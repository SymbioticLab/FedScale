import logging
import torch
import torch.nn as nn

MIN_FLOAT = torch.finfo(torch.float32).min / 100.0

class DLRM(nn.Module):
    def __init__(self,
                 args,
                 sync_mode=None):
        super(DLRM, self).__init__()
        self.dense_feature_dim = args.dense_feature_dim
        self.bot_layer_sizes = args.bot_layer_sizes
        self.sparse_feature_number = args.sparse_feature_number
        self.sparse_feature_dim = args.sparse_feature_dim
        self.top_layer_sizes = args.top_layer_sizes
        self.num_field = args.num_field

        self.bot_mlp = MLPLayer(input_shape=self.dense_feature_dim,
                                units_list=self.bot_layer_sizes,
                                last_action="relu")

        self.top_mlp = MLPLayer(input_shape=int(self.num_field * (self.num_field + 1) / 2) + self.sparse_feature_dim,
                                units_list=self.top_layer_sizes,last_action='sigmoid')
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=self.sparse_feature_dim)
            for size in self.sparse_feature_number
        ])

    def forward(self, sparse_inputs, dense_inputs):
        x = self.bot_mlp(dense_inputs)

        batch_size, d = x.shape
        sparse_embs = [self.embeddings[i](sparse_inputs[:, i]).view(-1, self.sparse_feature_dim) for i in range(sparse_inputs.shape[1])]

        T = torch.cat(sparse_embs + [x], axis=1).view(batch_size, -1, d)

        Z = torch.bmm(T, T.transpose(1, 2))
        Zflat = torch.triu(Z, diagonal=1) + torch.tril(torch.ones_like(Z) * MIN_FLOAT, diagonal=0)
        Zflat = Zflat.masked_select(Zflat > MIN_FLOAT).view(batch_size, -1)

        R = torch.cat([x] + [Zflat], axis=1)

        y = self.top_mlp(R)
        return y

class MLPLayer(nn.Module):
    def __init__(self, input_shape, units_list=None, l2=0.01, last_action=None):
        super(MLPLayer, self).__init__()

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.l2 = l2
        self.last_action = last_action
        self.mlp = nn.Sequential()

        for i in range(len(units_list)-1):
            self.mlp.add_module('dense_%d' % i, nn.Linear(units_list[i], units_list[i + 1]))
            if i != len(units_list) - 2 or last_action is not None:
                self.mlp.add_module('relu_%d' % i, nn.ReLU())
            self.mlp.add_module('norm_%d' % i, nn.BatchNorm1d(units_list[i + 1]))
        if last_action == 'sigmoid':
            self.mlp.add_module('sigmoid', nn.Sigmoid())

    def forward(self, inputs):
        return self.mlp(inputs)