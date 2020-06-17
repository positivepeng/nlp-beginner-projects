#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/13 16:01
@author: phil
"""

import torch.nn as nn
from torchcrf import CRF
import torch


class BiLSTM_CRF_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_tags, word_vectors=None, device="cpu"):
        super(BiLSTM_CRF_NER, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_dim, _weight=word_vectors).to(device)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True).to(device)
        self.hidden2tag = nn.Linear(hidden_size*2, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True).to(device)

    def forward(self, x, y, mask):
        emissions = self.get_emissions(x)
        loss = -self.crf(emissions=emissions, tags=y, mask=mask)
        return loss

    def predict(self, x, mask=None):
        emissions = self.get_emissions(x)
        preds = self.crf.decode(emissions, mask)
        return preds

    def get_emissions(self, x):
        batch_size, seq_len = x.shape
        embedded = self.embed(x)
        h0, c0 = torch.zeros(2, batch_size, self.hidden_size).to(self.device), torch.zeros(2, batch_size, self.hidden_size).to(self.device)
        lstm_out, (_, _) = self.lstm(embedded, (h0, c0))
        emissions = self.hidden2tag(lstm_out)
        return emissions
