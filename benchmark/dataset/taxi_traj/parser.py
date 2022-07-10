import collections
import csv
import os

import pandas


def parse_file(csv_file, is_train=True):
    dataset = pandas.read_csv(csv_file)
    client_meta = collections.defaultdict(list)

    for row in dataset.itertuples():
        client_meta[row.TAXI_ID].append(row)

    folder_name = 'train' if is_train else 'test'
    os.makedirs(folder_name, exist_ok=True)

    header = list(dataset.columns)
    for client_id in client_meta:
        client_data = [header] + client_meta[client_id]
        with open(f'./{folder_name}/{client_id}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(client_data)

    print(f"Parse {csv_file} done ...")

parse_file('train.csv')
parse_file('test.csv', False)
