#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/18 10:56
@author: phil
"""
import os
import pandas as pd
from torchtext import data


def prepare_data(dataset_path="../dataset/poetry"):
    file_path = os.path.join(dataset_path, "poetryFromTang.txt")
    target_path = os.path.join(dataset_path, "train.csv")
    if not os.path.exists(target_path):
        with open(file_path, encoding="utf-8") as f:
            lines = f.read().split("\n\n")
        lines = list(map(lambda x: x.replace("\n", ""), lines))
        df = pd.DataFrame()
        df["sent"] = lines
        df.to_csv(target_path, index=False, encoding='utf_8_sig')
    return target_path


def dataset2dataloader(dataset_path="../dataset/poetry", batch_size=32, debug=False):
    if debug:
        train_csv = os.path.join(dataset_path, "train_small.csv")
    else:
        train_csv = prepare_data(dataset_path)

    def tokenizer(text):
        return list(text)

    SENT = data.Field(sequential=True, tokenize=tokenizer, lower=False, init_token="<start>", eos_token="<end>")
    train, _ = data.TabularDataset.splits(path='', train=train_csv, validation=train_csv, format='csv',
                                          skip_header=True,
                                          fields=[('sent', SENT)])

    SENT.build_vocab(train)

    train_iter = data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.sent), shuffle=False)

    # 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
    # test_iter = data.Iterator(dataset=test, batch_size=128, train=False, sort=False, device=DEVICE)

    return train_iter, SENT.vocab


if __name__ == "__main__":
    train_iter, vocab = dataset2dataloader()
    for batch in train_iter:
        print(batch.sent.t())
        break
