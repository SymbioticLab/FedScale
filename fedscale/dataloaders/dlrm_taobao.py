import logging
import numpy as np
from torch.utils.data import Dataset


class Taobao(Dataset):
    def __init__(self, df):
        self.data_frame = df

        logging.info("init taobao...")

        self.sparse_features = [
            'userid', 'adgroup_id', 'cms_segid', 
            'cate_id', 'campaign_id', 'customer', 'brand'
        ]
        self.index_mappings = {
            feature: {v: k for k, v in enumerate(self.data_frame[feature].unique())} 
            for feature in self.sparse_features
        }

        dense_features = ['final_gender_code', 'pid', 'age_level','cms_group_id', 'shopping_level', 'occupation', 'new_user_class_level', 'time_stamp', 'price', 'pvalue_level']
        self.dense_features = dense_features
        for feature in self.dense_features:
            self.data_frame[feature] = self.data_frame[feature].astype(np.float32)
        
        self.targets = self.data_frame['clk'].values.astype(np.float32)
        logging.info("init taobao done...")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sparse_feature_values = [
            self.index_mappings[feature][self.data_frame.loc[idx, feature]]
            for feature in self.index_mappings
        ]
        sparse_x = np.array(sparse_feature_values, dtype=np.int64)
        dense_x_values = [
            self.data_frame.loc[idx, feature]
            for feature in self.dense_features
        ]
        dense_x = np.array(dense_x_values, dtype=np.float32)
        label = self.targets[idx].astype(np.float32)
    
        return dense_x, sparse_x, label