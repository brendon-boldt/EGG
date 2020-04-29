# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

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
        """
        x: hidden state of current step (of sender's message), [B, h]
        receiver_input: input data points (one real label and other distractors), [B, data_size, n_data]
        """
        _, data_size, _ = receiver_input.shape
        if self.method == "add":
            # TODO: seems like "add" not working. have to investigate.
            scores = torch.cat(
                (x.unsqueeze(1).repeat(1, data_size, 1), receiver_input), 2
            )
            # Additive attention: (B, s, h+d) --> (B, s, 1)
            scores = self.score(scores).squeeze(2)
        elif self.method == "mul":
            scores = self.score(x.unsqueeze(1))
            scores = torch.bmm(scores, receiver_input.transpose(1, 2)).squeeze(
                1
            )  # General attention: (B, 1, d) * (B, d, s) --> (B, 1, s)
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
                "Sender activation func.: [tanh|relu|leaky] supported at this moment"
            )

    def forward(self, x):
        x = self.fc1(x)
        return self.activation(x)


class SenderReceiverRnnGS(nn.Module):
    """See `core.gs_wrappers.SenderReceiverRnnGS`."""

    def __init__(
        self,
        sender,
        receiver,
        losses,
        vocab_size: int = None,
        vocab_prior: str = None,
        length_cost=0.0,
    ) -> None:
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts (TODO out of date)
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param length_cost: the penalty applied to Sender for each symbol produced
        """
        super(SenderReceiverRnnGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.losses = losses
        self.length_cost = length_cost
        self.vocab_prior = vocab_prior
        if vocab_prior is not None:
            assert vocab_size
        self._vocab_size = vocab_size
        self.reset_counts()

    def reset_counts(self) -> None:
        if self.vocab_prior == "dp":
            self.vocab_counts = torch.zeros(self._vocab_size)
        elif self.vocab_prior == "ema_dp":
            self.vocab_counts = torch.ones(self._vocab_size) / self._vocab_size
        else:
            self.vocab_counts = None

    def forward(self, sender_input, labels, receiver_input=None):
        message = self.sender(sender_input)
        receiver_output = self.receiver(message, receiver_input)

        loss = 0
        not_eosed_before = torch.ones(receiver_output.size(0)).to(
            receiver_output.device
        )
        expected_length = 0.0

        rest = {}
        z = 0.0
        vocab_counts = self.vocab_counts if self.training else None
        for step in range(receiver_output.size(1)):
            loss_args = {
                "sender_input": sender_input,
                "message": message[:, step, ...],
                "receiver_input": receiver_input,
                "receiver_output": receiver_output[:, step, ...],
                "labels": labels,
                "vocab_counts": vocab_counts,
                "vocab_prior": self.vocab_prior,
            }
            step_loss = torch.zeros(sender_input.shape[0])
            step_rest = {}
            for loss_fn in self.losses:
                term: Any = loss_fn(loss_args)
                if type(term) == tuple:
                    term, term_dict = term
                    step_rest = {**step_rest, **term_dict}
                step_loss += term

            # step_loss, step_rest = self.loss(
            #     sender_input,
            #     message[:, step, ...],
            #     receiver_input,
            #     receiver_output[:, step, ...],
            #     labels,
            #     vocab_counts=vocab_counts,
            #     vocab_prior=self.vocab_prior,
            # )
            eos_mask = message[:, step, 0]  # always eos == 0

            add_mask = eos_mask * not_eosed_before
            z += add_mask
            loss += step_loss * add_mask + self.length_cost * (1.0 + step) * add_mask
            expected_length += add_mask.detach() * (1.0 + step)

            for name, value in step_rest.items():
                rest[name] = value * add_mask + rest.get(name, 0.0)

            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += (
            step_loss * not_eosed_before
            + self.length_cost * (step + 1.0) * not_eosed_before
        )
        expected_length += (step + 1) * not_eosed_before

        z += not_eosed_before
        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_rest.items():
            rest[name] = value * not_eosed_before + rest.get(name, 0.0)
        for name, value in rest.items():
            rest[name] = value.mean()

        rest["mean_length"] = expected_length.mean()
        return loss.mean(), rest
