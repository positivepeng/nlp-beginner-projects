#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/18 11:05
@author: phil
"""

import torch.nn as nn
import torch


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, dropout=0.5):
        super(PoetryModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=vocab_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, init_hidden):
        # print(x.shape, init_hidden.shape)
        seq_out, hn = self.gru(x, init_hidden)
        output = self.out(seq_out)
        return output, hn

    def generate(self, x, stoi, poetry_type="begin", sent_num=4, max_len=15):
        init_hidden = torch.zeros(1, 1, self.hidden_size)
        output = []
        if poetry_type == "hidden head" and x.shape[1] != sent_num:
            print("ERROR：选择了藏头诗但是输入字的个数不等于诗的句子数")
            return

        hn = init_hidden
        for i in range(sent_num):
            if i == 0 and poetry_type == "begin":
                seq_out, hn = self.gru(x, hn)
                seq_out = seq_out[:, -1, :].unsqueeze(1)
                output.append(x)
            if poetry_type == "hidden head":
                seq_out, hn = self.gru(x[:, i, :].unsqueeze(1), hn)
                seq_out = seq_out[:, -1, :].unsqueeze(1)
                output.append(x[:, i, :].unsqueeze(1))
            for j in range(max_len):  # 每一句的最大长度
                # 上一个time step的输出
                _, topi = self.out(seq_out).data.topk(1)
                topi = topi.item()
                xi_from_output = torch.zeros(1, 1, x.shape[-1])
                xi_from_output[0][0][topi] = 1
                output.append(xi_from_output)
                seq_out, hn = self.gru(xi_from_output, hn)
                if topi == stoi["。"]:
                    break
        return output
