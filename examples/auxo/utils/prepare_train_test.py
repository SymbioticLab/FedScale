#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from collections import defaultdict
from os import path
import pandas as pd


def partition_by_client_unsort(file_path='train.csv'):
    print(f"Processing {file_path} into train and test sets.")

    clt_smp = defaultdict(list)
    title = None

    # Read CSV and partition by client
    with open(file_path, 'r') as fin:
        csv_reader = csv.reader(fin)
        for ind, row in enumerate(csv_reader):
            if ind == 0:
                title = row
            else:
                clt_smp[row[0]].append(ind - 1)

    # Check if output files already exist
    no_title = path.exists('train_by_clt.csv') or path.exists('test_by_clt.csv')

    # Initialize CSV writers for train and test sets
    with open('train_by_clt.csv', 'a') as write_partition_train_file, open('test_by_clt.csv',
                                                                           'a') as write_partition_test_file:
        writer_train = csv.writer(write_partition_train_file)
        writer_test = csv.writer(write_partition_test_file)

        if not no_title:
            writer_train.writerow(title)
            writer_test.writerow(title)

        # Read original CSV into a Pandas DataFrame
        print("Reading into pandas DataFrame.")
        df = pd.read_csv(file_path)

        cnt = 0
        clt_num = 0

        # Partition data and write to train and test CSV files
        for clt, samples in clt_smp.items():
            sample_num = len(samples)
            cnt += sample_num
            clt_num += 1

            for i, sample_idx in enumerate(samples):
                if i < sample_num * 0.8:
                    writer_train.writerow(list(df.loc[sample_idx].values))
                else:
                    writer_test.writerow(list(df.loc[sample_idx].values))

            if cnt % 10000 == 0:
                print(f"Wrote {cnt} samples.")
                print(f"Running average sample: {cnt / clt_num}")


if __name__ == "__main__":
    file_path = sys.argv[1]
    partition_by_client_unsort(file_path)
