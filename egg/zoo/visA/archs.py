# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ReinforceReceiver(nn.Module):
    def __init__(self, output_size, n_hidden):
        super(ReinforceReceiver, self).__init__()
        self.output = nn.Linear(n_hidden, output_size)

    def forward(self, x, receiver_input=None):
        logits = self.output(x).log_softmax(dim=1)
        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = logits.argmax(dim=1)
        log_prob = distr.log_prob(sample)

        return sample, log_prob, entropy


class Receiver(nn.Module):
    def __init__(self, n_data, n_hidden, method="mul"):
        super(Receiver, self).__init__()
        self.n_hidden = n_hidden
        self.method = method
        if method == "add":
            self.score = nn.Linear(n_hidden + n_data, 1)
        elif method == "mul":
            self.score = nn.Linear(n_hidden, n_data, bias=False)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x, receiver_input):
        '''
        x: hidden state of current step (of sender's message), [B, h]
        receiver_input: input data points (one real label and other distractors), [B, data_size, n_data]
        '''
        _, data_size, _ = receiver_input.shape
        if self.method == "add":
            # TODO: seems like "add" not working. have to investigate.
            scores = torch.cat((x.unsqueeze(1).repeat(
                1, data_size, 1), receiver_input), 2)
            # Additive attention: (B, s, h+d) --> (B, s, 1)
            scores = self.score(scores).squeeze(2)
        elif self.method == "mul":
            scores = self.score(x.unsqueeze(1))
            scores = torch.bmm(scores, receiver_input.transpose(1, 2)).squeeze(
                1)   # General attention: (B, 1, d) * (B, d, s) --> (B, 1, s)
        return self.sm(scores)


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features, activation="tanh"):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLu()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(
                "Sender activation func.: [tanh|relu|leaky] supported at this moment")

    def forward(self, x):
        x = self.fc1(x)
        return self.activation(x)
