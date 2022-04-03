#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:20:05 2021

@author: liujiachen
"""
from torch.utils.data import Dataset
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from tqdm import tqdm
import numpy as np
import nltk, os

# https://www.kaggle.com/eswarbabu88/toxic-comment-glove-logistic-regression
from collections import defaultdict

class AmazonReview_word2vec(Dataset):
    def __init__(self,  data_path, embedding_path = None, train = True ):
        
        file = 'train.csv' if train else 'test.csv'
        map_file = os.path.join(data_path, 'client_data_mapping', file)
        self.df = pd.read_csv(map_file, delimiter=',')
        # A reset reindexes from 1 to len(df), the shuffled df frames are sparse.
        self.df.reset_index(drop=True, inplace=True)
        
        self.client_mapping = defaultdict(list)
        self.data2clt = {}

        self.targets = []
        self.embeddings_index = {}
        self.embedding_file = embedding_path
        
        self.embedding(self.embedding_file)
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = stopwords.words('english')
        # client_id,data_path,label_name,label_id
        # initiate the (sample, client) pairs
        for  i, row in enumerate(self.df.itertuples()):
            (sample_id, client_id, data_path,label_name,label_id) = row
            client_id = int(client_id) - 1
            
            self.targets.append(float(label_name))
            self.client_mapping[client_id].append(sample_id)
            self.data2clt [i ] = client_id


    def __len__(self):
        return (len(self.df))
    
    def __getitem__(self, index):
        review = self.df.loc[index, 'data_path']
        # Classes start from 0.
        label = int(self.df.loc[index, 'label_name']) - 1
        return self.sent2vec(review), label
        
    def embedding(self, embedding_file):
        f = open(embedding_file, encoding="utf8")
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            try:
               coefs = np.asarray(values[1:], dtype='float32')
               self.embeddings_index[word] = coefs
            except ValueError:
               pass
        f.close()
        print('Found %s word vectors.' % len(self.embeddings_index))

    def sent2vec(self, s):
        words = str(s).lower()
        words = word_tokenize(words)
        words = [w for w in words if not w in self.stop_words]
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(self.embeddings_index[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        if type(v) != np.ndarray:
            return np.zeros(300)
        embed =  v / np.sqrt((v ** 2).sum())
        
        return  embed.astype(np.float32)
        # torch.from_numpy(embed)

    