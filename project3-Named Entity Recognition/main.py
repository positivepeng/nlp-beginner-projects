#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/13 16:15
@author: phil
"""

from dataloader import dataset2dataloader
from models import BiLSTM_CRF_NER
from torch.optim import Adam
import torch
from sklearn.metrics import f1_score
import numpy as np

if __name__ == "__main__":
    train_iter, val_iter, sent_vocab, tag_vocab = dataset2dataloader(batch_size=128)
    word_vectors = sent_vocab.vectors
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model = BiLSTM_CRF_NER(vocab_size=len(sent_vocab.stoi), embedding_dim=50, hidden_size=128, num_tags=len(tag_vocab.stoi), word_vectors=word_vectors, device=device)

    epoch = 2
    learning_rate = 0.01

    optimizer = Adam(model.parameters(), lr=learning_rate)

    for ep in range(epoch):
        model.train()
        for i, batch in enumerate(train_iter):
            x, y = batch.sent.t(), batch.tag.t()
            mask = (x != sent_vocab.stoi["<pad>"])
            optimizer.zero_grad()
            loss = model(x, y, mask)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"epoch:{ep}, iter:{i}, loss:{loss.item()}", end=" ")

        model.eval()
        train_accs = []
        preds, golds = [], []
        for i, batch in enumerate(train_iter):
            x, y = batch.sent.t(), batch.tag.t()
            mask = (x != sent_vocab.stoi["<pad>"])
            preds = model.predict(x, mask)
            right, total = 0, 0
            for pred, gold in zip(preds, y):
                right += np.sum(np.array(pred) == gold[:len(pred)].numpy())
                total += len(pred)
            train_accs.append(right*1.0/total)
        train_acc = np.array(train_accs).mean()

        val_accs = []
        for i, batch in enumerate(val_iter):
            x, y = batch.sent.t(), batch.tag.t()
            mask = (x != sent_vocab.stoi["<pad>"])
            preds = model.predict(x, mask)
            right, total = 0, 0
            for pred, gold in zip(preds, y):
                right += np.sum(np.array(pred) == gold[:len(pred)].numpy())
                total += len(pred)
            val_accs.append(right * 1.0 / total)
        val_acc = np.array(val_accs).mean()
        print("epoch %d train acc:%.2f, val acc:%.2f" % (epoch, train_acc, val_acc))
    test_sents = ["hello Peter , I am Phil .", "I am from European Union"]
    for sent in test_sents:
        ids = [sent_vocab.stoi[word] for word in sent.split(" ")]
        input_tensor = torch.tensor([ids])
        mask = input_tensor != sent_vocab.stoi["<pad>"]
        pred = model.predict(input_tensor, mask)
        print(sent, "-->", [tag_vocab.itos[tag_id] for tag_id in pred[0]])


