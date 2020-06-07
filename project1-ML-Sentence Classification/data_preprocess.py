#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/4/15 21:01
@author: phil
"""

import pandas as pd


def read_data(train_file="../dataset/kaggle-movie-review/train.tsv"):
    train_df = pd.read_csv(train_file, sep='\t')
    # test_df = pd.read_csv(test_file, sep="\t")
    return train_df["Phrase"].values, train_df["Sentiment"].values


if __name__ == "__main__":
    X_data, y_data = read_data()
    print("train size", len(X_data))
