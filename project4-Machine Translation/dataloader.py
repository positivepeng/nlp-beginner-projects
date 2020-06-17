#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/8 11:26
@author: phil
"""

# 参考吴恩达老师网易云深度学习课程作业
import os
import numpy as np
import torch
from faker import Faker
import random

from torch.nn import init
from tqdm import tqdm
from babel.dates import format_date
from torchtext import data
import pandas as pd
from sklearn.model_selection import train_test_split
fake = Faker()
Faker.seed(12345)
random.seed(12345)

# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

# change this if you want it to work with another language
LOCALES = ['en_US']


def load_date():
    """
        Loads some fake dates
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS), locale='en_US')
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt


def load_dataset(m):
    """
        Loads a dataset with m examples and vocabularies
        :m: the number of examples to generate
    """
    dataset = []
    for _ in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append([h, m])

    return dataset


def prepare_data(dataset_path=r"../dataset/date-normalization", dataset_size=10, debug=False):
    if debug:
        dataset_size = 10
        train_file = os.path.join(dataset_path, "train_samll.csv")
        eval_file = os.path.join(dataset_path, "eval_samll.csv")
    else:
        train_file = os.path.join(dataset_path, "train.csv")
        eval_file = os.path.join(dataset_path, "eval.csv")
    if not os.path.exists(train_file) and not os.path.exists(train_file):
        dataset = load_dataset(dataset_size)
        source, target = zip(*dataset)
        X_train, X_test, y_train, y_test = train_test_split(source, target, random_state=42, test_size=0.2)
        train_df = pd.DataFrame()
        train_df["source"], train_df["target"] = X_train, y_train
        eval_df = pd.DataFrame()
        eval_df["source"], eval_df["target"] = X_test, y_test
        train_df.to_csv(train_file, index=False)
        eval_df.to_csv(eval_file, index=False)
    return train_file, eval_file


def dataset2dataloader(dataset_path, batch_size=10, dataset_size=10, debug=False):
    train_csv, dev_csv = prepare_data(dataset_path, dataset_size=dataset_size, debug=debug)

    def tokenizer(text):
        return list(text)

    # 这里只是定义了数据格式
    SOURCE = data.Field(sequential=True, tokenize=tokenizer, lower=False)
    # 目标输出前后需加入特殊的标志符
    TARGET = data.Field(sequential=True, tokenize=tokenizer, lower=False, init_token="<start>", eos_token="<end>")
    train, val = data.TabularDataset.splits(
        path='', train=train_csv, validation=dev_csv, format='csv', skip_header=True,
        fields=[('source', SOURCE), ('target', TARGET)])

    SOURCE.build_vocab(train)
    TARGET.build_vocab(train)

    train_iter = data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.sent), shuffle=False)
    val_iter = data.BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.sent), shuffle=False)

    # 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
    # test_iter = data.Iterator(dataset=test, batch_size=128, train=False, sort=False, device=DEVICE)

    return train_iter, val_iter, SOURCE.vocab, TARGET.vocab