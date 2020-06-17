#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/13 11:58
@author: phil
"""
import os
import pandas as pd
import spacy
from torch.nn import init
from torchtext import data


def prepare_data(dataset_path, debug=False):
    train_file_path = os.path.join(dataset_path, "train.txt")
    dev_file_path = os.path.join(dataset_path, "dev.txt")

    def process_file(file_path, target_file_path):
        sents, tags = [], []
        with open(file_path, "r") as f:
            lines = f.readlines()
            sent, tag = [], []
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    sents.append(" ".join(sent))
                    tags.append(" ".join(tag))
                    sent, tag = [], []
                else:
                    splited = line.split(" ")
                    sent.append(splited[0])
                    tag.append(splited[-1])
            if len(sent) != 0:
                sents.append(" ".join(sent))
                tags.append(" ".join(tag))
        df = pd.DataFrame()
        df["sent"] = sents if not debug else sents[:100]
        df["tag"] = tags if not debug else tags[:100]
        df.to_csv(target_file_path, index=False)

    train_csv = os.path.join(dataset_path, "train.csv") if not debug else os.path.join(dataset_path, "train_small.csv")
    dev_csv = os.path.join(dataset_path, "dev.csv") if not debug else os.path.join(dataset_path, "train_dev.csv")

    if not os.path.exists(train_csv):
        process_file(train_file_path, train_csv)
        process_file(dev_file_path, dev_csv)

    return train_csv, dev_csv


def dataset2dataloader(dataset_path="../dataset/conll2003-IOB", batch_size=3, debug=False):
    train_csv, dev_csv = prepare_data(dataset_path, debug=debug)

    def tokenizer(text):
        return text.split(" ")

    # 这里只是定义了数据格式
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=False)
    TAG = data.Field(sequential=True, tokenize=tokenizer, lower=False)
    train, val = data.TabularDataset.splits(
        path='', train=train_csv, validation=dev_csv, format='csv', skip_header=True,
        fields=[('sent', TEXT), ('tag', TAG)])

    TEXT.build_vocab(train, vectors='glove.6B.50d')  # , max_size=30000)
    TAG.build_vocab(val)

    # 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
    TEXT.vocab.vectors.unk_init = init.xavier_uniform

    DEVICE = "cpu"
    train_iter = data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.sent), device=DEVICE)
    val_iter = data.BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.sent), device=DEVICE)

    # 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
    # test_iter = data.Iterator(dataset=test, batch_size=128, train=False, sort=False, device=DEVICE)

    return train_iter, val_iter, TEXT.vocab, TAG.vocab


if __name__ == "__main__":
    # train_csv, dev_csv = prepare_data(dataset_path="../dataset/conll2003-IOB")
    train_iter, val_iter, sent_vocab, tag_vocab = dataset2dataloader(dataset_path="../dataset/conll2003-IOB", debug=True)
    word_vectors = sent_vocab.vectors

    for batch in train_iter:
        print(batch.sent.shape, batch.tag.shape)
        break