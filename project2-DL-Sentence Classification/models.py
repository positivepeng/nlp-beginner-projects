#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/5/15 22:23
@author: phil
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_of_class, weights=None, rnn_type="RNN"):
        super(TextRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_of_class = num_of_class
        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type

        if weights is not None:
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, _weight=weights)
        else:
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        if rnn_type == "RNN":
            self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
            self.hidden2label = nn.Linear(hidden_size, num_of_class)
        elif rnn_type == "LSTM":
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            self.hidden2label = nn.Linear(hidden_size*2, num_of_class)

    def forward(self, input_sents):
        # input_sents (batch_size, seq_len)
        batch_size, seq_len = input_sents.shape
        # (batch_size, seq_len, embedding_dim)
        embed_out = self.embed(input_sents)

        if self.rnn_type == "RNN":
            h0 = torch.randn(1, batch_size, self.hidden_size)
            _, hn = self.rnn(embed_out, h0)
        elif self.rnn_type == "LSTM":
            h0, c0 = torch.randn(2, batch_size, self.hidden_size), torch.randn(2, batch_size, self.hidden_size)
            output, (hn, _) = self.lstm(embed_out, (h0, c0))

        logits = self.hidden2label(hn).squeeze(0)

        return logits


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_of_class, embedding_vectors=None, kernel_num=100, kerner_size=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        if embedding_vectors is None:
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, _weight=embedding_vectors)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embedding_dim)) for K in kerner_size])
        self.dropout = nn.Dropout(dropout)
        self.feature2label = nn.Linear(3*kernel_num, num_of_class)

    def forward(self, x):
        # x shape (batch_size, seq_len)
        embed_out = self.embed(x).unsqueeze(1)
        conv_out = [F.relu(conv(embed_out)).squeeze(3) for conv in self.convs]

        pool_out = [F.max_pool1d(block, block.size(2)).squeeze(2) for block in conv_out]

        pool_out = torch.cat(pool_out, 1)

        logits = self.feature2label(pool_out)

        return logits


if __name__ == "__main__":
    model = TextCNN(vocab_size=10, embedding_dim=10, num_of_class=10)
    x = torch.randint(10, (10, 20))
    logits = model.forward(x)

