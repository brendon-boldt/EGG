# fmt: off
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import sys
import argparse
import contextlib
import math
import logging
from typing import Dict, Any, Callable

import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from torch.utils.data import DataLoader

from egg.zoo.visA.features import VisaDataset, InaDataset, GroupedInaDataset
from egg.zoo.visA import archs
from egg.zoo.visA import callbacks
from egg.zoo.visA.trainers import Trainer


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='ina',
                        help='Name of the dataset to use {ina, visa}')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the dataset')
    parser.add_argument('--valid_prop', type=float, default=0.2,
                        help='Proportion of dataset to use for validation')
    parser.add_argument('--n_distractors', type=int, default=4,
                        help='Number of distractors for receiver to see')
    parser.add_argument('--dump_data', type=str, default=None,
                        help='Path to the data for which to produce output information')
    parser.add_argument('--dump_output', type=str, default=None,
                        help='Path for dumping output information')

    parser.add_argument('--examples_per_epoch', type=int, default=-1,
                        help='Number of batches per epoch (default: size of training set)')

    parser.add_argument('--sender_hidden', type=int, default=20,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=20,
                        help='Size of the hidden layer of Receiver (default: 10)')

    parser.add_argument('--sender_embedding', type=int, default=20,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=20,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--sender_cell', type=str, default='gru',
                        help='Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='gru',
                        help='Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)')
    parser.add_argument('--sender_layers', type=int, default=1,
                        help="Number of layers in Sender's RNN (default: 1)")
    parser.add_argument('--receiver_layers', type=int, default=1,
                        help="Number of layers in Receiver's RNN (default: 1)")

    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-2,
                        help='The entropy regularisation coefficient for Sender (default: 1e-2)')
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-2,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-2)')

    parser.add_argument('--sender_lr', type=float, default=1e-1,
                        help="Learning rate for Sender's parameters (default: 1e-1)")
    parser.add_argument('--receiver_lr', type=float, default=1e-1,
                        help="Learning rate for Receiver's parameters (default: 1e-1)")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender (default: 1.0)")
    parser.add_argument('--train_mode', type=str, default='gs',
                        help="Selects whether GumbelSoftmax or Reinforce is used"
                             "(default: gs)")
    parser.add_argument('--toposim', type=bool, default=False,
                        help="boolean for measureing topological similarity (default: False)")
    parser.add_argument('--print_train', type=bool, default=False,
                        help="boolean specify whether to print Train res (default: False)")
    parser.add_argument('--print_test', type=bool, default=True,
                        help="boolean specify whether to print Test res (default: True)")
    parser.add_argument('--toposim_embed', type=bool, default=True,
                        help=("If true, use cosine distance of embeddings of messages, "
                              "otherwise use Levenshtein distance."))

    parser.add_argument('--n_classes', type=int, default=None,
                        help='Number of classes for Receiver to output. If not set, is automatically deduced from '
                             'the training set')

    parser.add_argument('--force_eos', action='store_true', default=False,
                        help="When set, forces that the last symbol of the message is EOS (default: False)")

    parser.add_argument('--early_stopping_thr', type=float, default=0.98,
                        help="Early stopping threshold on accuracy (defautl: 0.98)")
    args = core.init(parser)
    return args


def dump(game, dataset, device, is_gs):
    sender_inputs, messages, _, receiver_outputs, labels = \
        core.dump_sender_receiver(
            game, dataset, gs=is_gs, device=device, variable_length=True)

    for sender_input, message, receiver_output, label \
            in zip(sender_inputs, messages, receiver_outputs, labels):
        sender_input = ' '.join(map(str, sender_input.tolist()))
        message = ' '.join(map(str, message.tolist()))
        if is_gs:
            receiver_output = receiver_output.argmax()
        logging.info(f'{sender_input};{message};{receiver_output};{label.item()}')
# fmt: on


def dirichlet_process_prior(alpha=2.0, lambda_0=1e-4) -> Callable[..., torch.Tensor]:
    def f(kwargs: Dict[str, Any]) -> torch.Tensor:
        # During eval
        # TODO properly handle eval vs. train
        if kwargs["vocab_counts"] is None:
            return torch.zeros(1)
        new_v_counts = kwargs["message"].sum(0)
        kwargs["vocab_counts"] += kwargs["message"].sum(0).detach()
        vocab_loss = (
            -new_v_counts
            * (new_v_counts / (alpha - 1 + kwargs["vocab_counts"].sum())).log()
        )
        return lambda_0 * vocab_loss.sum()

    return f


def ema_dirichlet_process_prior(
    alpha=2.0, lambda_0=1e-4, gamma=0.9
) -> Callable[..., torch.Tensor]:
    def f(kwargs: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError()
        gamma = 0.9
        # v_counts = gamma * v_counts +  (1 - gamma) * _message.sum(0).detach()

    return f


def differentiable_loss(kwargs: Dict[str, Any]) -> torch.Tensor:
    # _sender_input, _message, _receiver_input, receiver_output, labels
    res_dict = {}
    labels = kwargs["labels"].squeeze(1)
    acc = (kwargs["receiver_output"].argmax(dim=1) == labels).detach().float()
    res_dict["acc"] = acc
    loss = F.cross_entropy(kwargs["receiver_output"], labels, reduction="none")
    return loss, res_dict


# fmt: off
def non_differentiable_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    labels = labels.squeeze(1)
    acc = (receiver_output == labels).detach().float()
    return -acc, {'acc': acc}


def build_model(opts, train_loader, dump_loader) -> Any:
    n_features = train_loader.dataset.get_n_features(
    ) if train_loader else dump_loader.dataset.get_n_features()

    if opts.n_classes is not None:
        receiver_outputs = opts.n_classes
    else:
        receiver_outputs = train_loader.dataset.get_output_max() + 1 if train_loader else \
            dump_loader.dataset.get_output_max() + 1

    sender = archs.Sender(n_hidden=opts.sender_hidden, n_features=n_features)

    receiver = archs.Receiver(n_data=n_features,
                        n_hidden=opts.receiver_hidden)
    loss = differentiable_loss
    # TODO: implement non_differentiable_loss for rf?
    # if opts.train_mode.lower() == 'gs':
    #     loss = differentiable_loss
    #     receiver = Receiver(output_size=receiver_outputs, n_hidden=opts.receiver_hidden)
    # else:
    #     loss = non_differentiable_loss
    #     receiver = ReinforceReceiver(output_size=receiver_outputs, n_hidden=opts.receiver_hidden)

    return sender, receiver, loss

optimizers = {'adam': torch.optim.Adam,
             'sgd': torch.optim.SGD,
             'adagrad': torch.optim.Adagrad}

# fmt: on
def run_game(opts: argparse.Namespace) -> Dict[str, Any]:
    logging.info(f"Launching game with parameters: {opts}")
    logs: Dict[str, Any] = {"opts": opts}

    device = torch.device("cuda" if opts.cuda else "cpu")

    if opts.data_path is None:
        # TODO This can be specfied by the argparser...
        raise ValueError("--data_path must be supplied")

    # TODO And so can the limited options for the dataset
    if opts.data_set == "gina":
        train_dataset, validation_dataset = GroupedInaDataset.from_file(
            opts.data_path, opts.n_distractors
        )
    else:
        if opts.data_set == "visa":
            whole_dataset = VisaDataset.from_file(opts.data_path, opts.n_distractors)
        elif opts.data_set == "ina":
            whole_dataset = InaDataset.from_file(opts.data_path, opts.n_distractors)
        validation_dataset, train_dataset = whole_dataset.valid_train_split(
            opts.valid_prop
        )
    if opts.examples_per_epoch > 0:
        train_dataset.n_repeats = math.ceil(
            opts.examples_per_epoch / len(train_dataset)
        )
    logging.info(f"Examples per epoch: {len(train_dataset)}")
    validation_loader = DataLoader(
        validation_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=0
    )
    train_loader = DataLoader(
        train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=0
    )

    dump_loader = None

    assert train_loader or dump_loader, "Either training or dump data must be specified"
    sender, receiver, loss = build_model(opts, train_loader, dump_loader)

    if opts.train_mode.lower() == "rf":
        sender = core.RnnSenderReinforce(
            sender,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            force_eos=opts.force_eos,
            num_layers=opts.sender_layers,
        )
        receiver = core.RnnReceiverDeterministic(
            receiver,
            opts.vocab_size,
            opts.receiver_embedding,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
            num_layers=opts.receiver_layers,
        )

        game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            differentiable_loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=opts.receiver_entropy_coeff,
        )
    elif opts.train_mode.lower() == "gs" or opts.train_mode.lower() == "stgs":
        straight_through = opts.train_mode.lower() == "stgs"
        sender = core.RnnSenderGS(
            sender,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.temperature,
            force_eos=opts.force_eos,
            straight_through=straight_through,
        )

        receiver = core.RnnReceiverGS(
            receiver,
            opts.vocab_size,
            opts.receiver_embedding,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
        )

        losses = [differentiable_loss]
        if opts.vocab_prior == "dp":
            losses.append(
                dirichlet_process_prior(alpha=opts.dp_alpha, lambda_0=opts.dp_lambda_0)
            )
        elif opts.vocab_prior == "ema_dp":
            losses.append(
                ema_dirichlet_process_prior(
                    alpha=opts.dp_alpha, lambda_0=opts.dp_lambda_0, gamma=opts.dp_gamma
                )
            )
        game = archs.SenderReceiverRnnGS(
            sender,
            receiver,
            losses,
            vocab_prior=opts.vocab_prior,
            vocab_size=opts.vocab_size,
        )
    else:
        raise NotImplementedError(f"Unknown training mode, {opts.mode}")

    # optimizer = core.build_optimizer(game.parameters())
    optimizer = optimizers[opts.optimizer](game.parameters(), lr=opts.lr)

    # early_stopper = core.EarlyStopperAccuracy(threshold=opts.early_stopping_thr, field_name="acc", validation=True)
    metric_logger = callbacks.MetricLogger()
    callback_list = [
        callbacks.VocabCountsReset(game),
        callbacks.ToposimCallback(
            validation_loader, train_loader, sender, use_embeddings=opts.toposim_embed
        ),
        callbacks.ConsoleLogger(print_train_loss=True, print_test_loss=True),
        # It is important that this logger is last so it picks up any post-hoc metrics
        metric_logger,
    ]
    trainer = Trainer(
        game=game,
        opts=opts,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callback_list,
    )

    trainer.train(n_epochs=opts.n_epochs)
    if opts.checkpoint_dir:
        trainer.save_checkpoint()

    core.close()
    # Just in case the opts object has been edited
    logs["post_opts"] = opts
    metric_logs = metric_logger.get_finalized_logs()
    return post_process_logs({**metric_logs, **logs}, len(train_dataset))


def post_process_logs(logs: Dict[str, Any], examples_per_epoch: int) -> Dict[str, Any]:
    logs["examples_per_epoch"] = examples_per_epoch
    logs["objective"] = -logs["valid"]["acc"].max()
    return logs


if __name__ == "__main__":
    opts = get_params()
    run_game(opts)
