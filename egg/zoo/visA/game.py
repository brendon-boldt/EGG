# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import sys
import argparse
import contextlib
import math

import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from torch.utils.data import DataLoader
from hyperopt import hp, tpe, fmin, Trials

from egg.zoo.visA.features import VisaDataset, InaDataset, GroupedInaDataset
from egg.zoo.visA.archs import Sender, Receiver, ReinforceReceiver
from egg.zoo.visA.callbacks import ToposimCallback, MinValLossCallback, calculate_toposim


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
        print(f'{sender_input};{message};{receiver_output};{label.item()}')


def differentiable_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    res_dict = {}
    labels = labels.squeeze(1)
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    res_dict['acc'] = acc
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, res_dict


def non_differentiable_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    labels = labels.squeeze(1)
    acc = (receiver_output == labels).detach().float()
    return -acc, {'acc': acc}


def build_model(opts, train_loader, dump_loader):
    n_features = train_loader.dataset.get_n_features(
    ) if train_loader else dump_loader.dataset.get_n_features()

    if opts.n_classes is not None:
        receiver_outputs = opts.n_classes
    else:
        receiver_outputs = train_loader.dataset.get_output_max() + 1 if train_loader else \
            dump_loader.dataset.get_output_max() + 1

    sender = Sender(n_hidden=opts.sender_hidden, n_features=n_features)

    receiver = Receiver(n_data=n_features,
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


def run_model(args):
    device = torch.device("cuda" if opts.cuda else "cpu")

    if opts.data_path is None:
        # TODO This can be specfied by the argparser...
        raise ValueError("--data_path must be supplied")

    # TODO And so can the limited options for the dataset
    if opts.data_set == 'gina':
        train_dataset, validation_dataset = GroupedInaDataset.from_file(
            opts.data_path,
            opts.n_distractors,
        )
    else:
        if opts.data_set == 'visa':
            whole_dataset = VisaDataset.from_file(
                opts.data_path, opts.n_distractors)
        elif opts.data_set == 'ina':
            whole_dataset = InaDataset.from_file(
                opts.data_path, opts.n_distractors)
        validation_dataset, train_dataset = whole_dataset.valid_train_split(
            opts.valid_prop)
    if opts.examples_per_epoch > 0:
        train_dataset.n_repeats = math.ceil(
            opts.examples_per_epoch / len(train_dataset)
        )
    print(f"Examples per epoch: {len(train_dataset)}")
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=1
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=1
    )

    dump_loader = None

    assert train_loader or dump_loader, 'Either training or dump data must be specified'
    sender, receiver, loss = build_model(opts, train_loader, dump_loader)

    if opts.train_mode.lower() == 'rf':
        sender = core.RnnSenderReinforce(sender,
                                         opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                         cell=opts.sender_cell, max_len=opts.max_len, force_eos=opts.force_eos,
                                         num_layers=opts.sender_layers)
        receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_embedding,
                                                 opts.receiver_hidden, cell=opts.receiver_cell,
                                                 num_layers=opts.receiver_layers)

        game = core.SenderReceiverRnnReinforce(sender, receiver, differentiable_loss, sender_entropy_coeff=opts.sender_entropy_coeff,
                                               receiver_entropy_coeff=opts.receiver_entropy_coeff)
    elif opts.train_mode.lower() == 'gs':
        sender = core.RnnSenderGS(sender, opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                  cell=opts.sender_cell, max_len=opts.max_len, temperature=opts.temperature,
                                  force_eos=opts.force_eos)

        receiver = core.RnnReceiverGS(receiver, opts.vocab_size, opts.receiver_embedding,
                                      opts.receiver_hidden, cell=opts.receiver_cell)

        game = core.SenderReceiverRnnGS(sender, receiver, differentiable_loss)
    else:
        raise NotImplementedError(f'Unknown training mode, {opts.mode}')

    optimizer = core.build_optimizer(game.parameters())
    # early_stopper = core.EarlyStopperAccuracy(threshold=opts.early_stopping_thr, field_name="acc", validation=True)
    callbacks = [
        ToposimCallback(validation_loader, train_loader, sender,
                        use_embeddings=opts.toposim_embed),
        core.ConsoleLogger(print_train_loss=opts.print_train,
                           print_test_loss=opts.print_test),
        MinValLossCallback(),
    ]
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=validation_loader, callbacks=callbacks)

    res = trainer.train(n_epochs=opts.n_epochs)
    if opts.checkpoint_dir:
        trainer.save_checkpoint()
    res = [i for i in res if i != None][0]

    return res


if __name__ == "__main__":
    opts = get_params()
    search_space = {
        'train_mode': hp.choice('train_mode', ['gs', 'rf']),
        'max_lens': hp.uniform('max_lens', 1, 10),
        'vocab': hp.uniform('vocab', 10, 200),
        'lr': hp.uniform('lr', 0.00001, 0.0001),
        'n_epochs': hp.uniform('n_epochs', 500, 1000),
        'sender_hidden': hp.uniform('sender_hidden', 10, 100),
        'receiver_hidden': hp.uniform('receiver_hidden', 10, 100),
    }

    print(f'Initial parameters: {opts}')
    print(f'Bayesian optimization over: {search_space}')

    hypopt_trials = Trials()
    best_params = fmin(run_model, search_space, algo=tpe.suggest,
                       max_evals=100, trials=hypopt_trials)
    print(best_params)
    core.close()
