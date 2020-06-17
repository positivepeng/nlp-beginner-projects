#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/8 11:36
@author: phil
"""
from keras.utils import to_categorical

from dataloader import load_dataset, dataset2dataloader
from models import SimpleNMT
from torch import optim
import torch.nn as nn
import torch
import numpy as np
from pprint import pprint
from tqdm import tqdm

if __name__ == "__main__":
    epoch = 500
    learning_rate = 0.001
    hidden_size = 64
    batch_size = 10

    train_iter, val_iter, source_vocab, target_vocab = dataset2dataloader(dataset_path=r"../dataset/date-normalization",
                                                                          batch_size=batch_size, dataset_size=10000, debug=True)
    source_vocab_size = len(source_vocab.stoi)
    target_vocab_size = len(target_vocab.stoi)

    # print(target_vocab.stoi)

    Tx, Ty = 25, 10  # 最大长度

    model = SimpleNMT(in_vocab_size=source_vocab_size, out_vocab_size=target_vocab_size, in_hidden_size=hidden_size,
                      out_hidden_size=hidden_size, output_size=target_vocab_size, with_attention=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    embed_layer1 = nn.Embedding(source_vocab_size, source_vocab_size,
                                _weight=torch.from_numpy(np.eye(source_vocab_size)))
    embed_layer2 = nn.Embedding(target_vocab_size, target_vocab_size,
                                _weight=torch.from_numpy(np.eye(target_vocab_size)))

    model.train()
    for ep in range(epoch):
        epoch_loss = 0
        for batch in train_iter:
            optimizer.zero_grad()
            Xin, Yin, Yout = batch.source.t().long(), batch.target.t()[:, :-1].long(), batch.target.t()[:, 1:]
            batch_size = len(Xin)
            init_hidden = torch.zeros(1, batch_size, hidden_size)
            # if ep == epoch - 1:
            #     print(Yout)
            Xin = embed_layer1(Xin).float()
            Yin = embed_layer2(Yin).float()
            logits = model(Xin, init_hidden, Yin)
            loss = criterion(logits.view(-1, logits.shape[-1]), Yout.flatten())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        if ep % (epoch // 10) == 0:
            print("loss", epoch_loss)

    # 测试训练集输出是否正确
    # for batch in train_iter:
    #     # print(batch.source.t())
    #     print(batch.target.t()[:, 1:])
    # print("finish")
    # init_hidden = torch.zeros(1, batch_size, hidden_size)
    # logits = model(Xin, init_hidden, Yin)
    # print(logits.argmax(-1))

    sents_for_large = ["monday may 7 1983", "19 march 1998", "18 jul 2008", "9/10/70", "thursday january 1 1981",
                       "thursday january 26 2015", "saturday april 18 1990", "sunday may 12 1988"]
    sents = ["monday march 7 1983", "9 may 1998", "thursday january 26 1995", "9/10/70"]


    def translate(model, sents):
        X = []
        for sent in sents:
            X.append(list(map(lambda x: source_vocab[x], list(sent))) + [source_vocab["<pad>"]] * (Tx - len(sent)))
        Xoh = torch.from_numpy(np.array(list(map(lambda x: to_categorical(x, num_classes=source_vocab_size), X))))
        encoder_init_hidden = torch.zeros(1, len(X), hidden_size)
        preds = model(Xoh, encoder_init_hidden, decoder_input=None, out_word2index=target_vocab.stoi,
                      out_index2word=target_vocab.itos, max_len=Ty, out_size=target_vocab_size)
        for gold, pred in zip(sents, preds):
            print(gold, "-->", "".join(pred))


    translate(model, sents)

    """ 不使用 attention
    dataset_size : 10000
    loss 940.5139790773392
    loss 151.68325132876635
    loss 17.91189043689519
    loss 8.461621267197188
    loss 0.4571912245155545
    loss 4.067497536438168
    loss 0.02432645454427984
    loss 0.022933890589229122
    loss 1.740354736426525
    loss 2.7019595313686295
    monday may 7 1983 --> 1983-05-07
    19 march 1998 --> 1998-03-19
    18 jul 2008 --> 2008-07-18
    9/10/70 --> 1970-09-10
    thursday january 1 1981 --> 1981-01-01
    thursday january 26 2015 --> 2015-01-26
    saturday april 18 1990 --> 1990-04-18
    sunday may 12 1988 --> 1988-05-12
    """

    """使用attention
    loss 870.4544065594673
    loss 65.41884177550673
    loss 53.339022306521656
    loss 0.08635593753569992
    loss 0.057157438381182146
    loss 0.0006471980702968949
    loss 0.09261544834953384
    loss 0.000922315769471993
    loss 0.00961817828419953
    loss 0.06814217135979561
    monday may 7 1983 --> 1983-05-07
    19 march 1998 --> 1998-03-19
    18 jul 2008 --> 2008-07-18
    9/10/70 --> 1970-09-10
    thursday january 1 1981 --> 1981-01-01
    thursday january 26 2015 --> 2015-01-26
    saturday april 18 1990 --> 1990-04-18
    sunday may 12 1988 --> 1988-05-12
    """
