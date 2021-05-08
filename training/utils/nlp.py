# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    AlbertConfig,
    AlbertForMaskedLM,
    AlbertTokenizer,
    get_linear_schedule_with_warmup,
    AutoModelWithLMHead,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
    "albert": (AlbertConfig, AlbertForMaskedLM, AlbertTokenizer),
}


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)
        is_folder = True if args.data_mapfile is not None else False

        if is_folder == False:
            assert os.path.isfile(file_path)
            directory, filename = os.path.split(file_path)
            cached_features_file = os.path.join(
                    directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
                )   
        else:
            directory = file_path
            cached_features_file = os.path.join(
                    directory, args.model_type + "_cached_lm_" + str(block_size)
                )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
                self.client_mapping = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            self.client_mapping = {}
            sample_id = -1
            user_id = -1

            if is_folder == False:
                files = [file_path]
            else:
                files = [os.path.join(file_path, entry.name) for entry in os.scandir(file_path) if '_cached_lm_' not in entry.name]

            # make sure files are ordered
            files = sorted(files)
            
            for file in files:
                with open(file, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                if len(tokenized_text) > 0:
                    user_id += 1
                    self.client_mapping[user_id] = []

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    sample_id += 1
                    self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
                    self.client_mapping[user_id].append(sample_id)

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=-1)
                pickle.dump(self.client_mapping, handle, protocol=-1)

        self.data = self.examples
        self.targets = [0 for i in range(len(self.data))]
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        self.data = self.examples
        self.targets = [0 for i in range(len(self.data))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.data_mapfile is None:
        file_path = args.eval_data_file if evaluate else args.train_data_file
    else:
        file_path = os.path.join(args.data_dir, 'test') if evaluate else os.path.join(args.data_dir, 'train')

    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)

def invertBool(list):
    return abs(list-1)
    # size = list.size()
    # temp = torch.zeros(size, dtype=torch.bool).cuda()
    # for i in range(size[0]):
    #     for j in range(size[1]):
    #         temp[i][j] = False if list[i][j] else True
    # return temp

def toBool(list):
    # size = list.size()
    # temp = torch.zeros(size, dtype=torch.bool)
    # for i in range(size[0]):
    #     for j in range(size[1]):
    #         temp[i][j] = True if list[i][j] else False
    return list

def boolAnd(list1, list2):
    size = list1.size()
    temp = torch.zeros(size, dtype=torch.bool)
    for i in range(size[0]):
        for j in range(size[1]):
            temp[i][j] = list1[i][j] & list2[i][j]
    return temp

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args, device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone(device=device)
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability).to(device=device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.uint8, device=device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.tensor(torch.bernoulli(probability_matrix), dtype=torch.uint8).detach().to(device=device)
    labels[toBool(~masked_indices)] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.tensor(torch.bernoulli(torch.full(labels.shape, 0.8)), dtype=torch.uint8, device=device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    #indices_random = boolAnd(boolAnd(toBool(torch.bernoulli(torch.full(labels.shape, 0.5))), toBool(masked_indices)), invertBool(indices_replaced))
    indices_random = torch.tensor(torch.bernoulli(torch.full(labels.shape, 0.5)), dtype=torch.uint8, device=device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    bool_indices_random = toBool(indices_random)
    inputs[bool_indices_random] = random_words[bool_indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


