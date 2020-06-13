#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/12 15:25
@author: phil
"""

import pandas as pd
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


def prepare_data(dataset_path, sent_col_name, label_col_name):
    """ 读出tsv中的句子和标签 """
    file_path = os.path.join(dataset_path, "train.tsv")
    data = pd.read_csv(file_path, sep="\t")
    X = data[sent_col_name].values
    y = data[label_col_name].values
    return X, y


class Language:
    """ 根据句子列表建立词典并将单词列表转换为数值型表示 """
    def __init__(self):
        self.word2id = {}
        self.id2word = {}

    def fit(self, sent_list):
        vocab = set()
        for sent in sent_list:
            vocab.update(sent.split(" "))
        word_list = ["<pad>", "<unk>"] + list(vocab)
        self.word2id = {word: i for i, word in enumerate(word_list)}
        self.id2word = {i: word for i, word in enumerate(word_list)}

    def transform(self, sent_list, reverse=False):
        sent_list_id = []
        word_mapper = self.word2id if not reverse else self.id2word
        unk = self.word2id["<unk>"] if not reverse else None
        for sent in sent_list:
            sent_id = list(map(lambda x: word_mapper.get(x, unk), sent.split(" ") if not reverse else sent))
            sent_list_id.append(sent_id)
        return sent_list_id


class ClsDataset(Dataset):
    """ 文本分类数据集 """
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels

    def __getitem__(self, item):
        return self.sents[item], self.labels[item]

    def __len__(self):
        return len(self.sents)


def collate_fn(batch_data):
    """ 自定义一个batch里面的数据的组织方式 """
    batch_data.sort(key=lambda data_pair: len(data_pair[0]), reverse=True)

    sents, labels = zip(*batch_data)
    sents_len = [len(sent) for sent in sents]
    sents = [torch.LongTensor(sent) for sent in sents]
    padded_sents = pad_sequence(sents, batch_first=True, padding_value=0)

    return torch.LongTensor(padded_sents), torch.LongTensor(labels),  torch.FloatTensor(sents_len)


def get_wordvec(word2id, vec_file_path, vec_dim=50):
    """ 读出txt文件的预训练词向量 """
    print("开始加载词向量")
    word_vectors = torch.nn.init.xavier_uniform_(torch.empty(len(word2id), vec_dim))
    word_vectors[0, :] = 0  # <pad>
    found = 0
    with open(vec_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            splited = line.split(" ")
            if splited[0] in word2id:
                found += 1
                word_vectors[word2id[splited[0]]] = torch.tensor(list(map(lambda x: float(x), splited[1:])))
            if found == len(word2id) - 1:  # 允许<unk>找不到
                break
    print("总共 %d个词，其中%d个找到了对应的词向量" % (len(word2id), found))
    return word_vectors.float()


def make_dataloader(dataset_path="../dataset/kaggle-movie-review", sent_col_name="Phrase", label_col_name="Sentiment", batch_size=32, vec_file_path="./.vector_cache/glove.6B.50d.txt", debug=False):
    # X, y = prepare_datapairs(dataset_path="../dataset/imdb", sent_col_name="review", label_col_name="sentiment")
    X, y = prepare_data(dataset_path=dataset_path, sent_col_name=sent_col_name, label_col_name=label_col_name)

    if debug:
        X, y = X[:100], y[:100]

    X_language = Language()
    X_language.fit(X)
    X = X_language.transform(X)

    word_vectors = get_wordvec(X_language.word2id, vec_file_path=vec_file_path, vec_dim=50)
    # 总共 18229个词，其中12769个找到了对应的词向量 word_vectors = get_wordvec(X_language.word2id,
    # vec_file_path=r"F:\NLP-pretrained-model\glove.twitter.27B\glove.twitter.27B.50d.txt", vec_dim=50)

    # 测试
    # print(X[:2])
    # X_id = X_language.transform(X[:2])
    # print(X_language.transform(X_id, reverse=True))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    cls_train_dataset, cls_val_dataset = ClsDataset(X_train, y_train), ClsDataset(X_val, y_val)
    cls_train_dataloader = DataLoader(cls_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    cls_val_dataloader = DataLoader(cls_val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return cls_train_dataloader, cls_val_dataloader, word_vectors, X_language


if __name__ == "__main__":
    cls_train_dataloader, cls_val_dataloader, word_vectors, X_language = make_dataloader(debug=True, batch_size=10)
    for batch in cls_train_dataloader:
        X, y, lens = batch
        print(X.shape, y.shape)
        break
