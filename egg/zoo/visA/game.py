# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import sys
import argparse
import contextlib
import warnings
import math

import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.zoo.visA.features import VisaDataset
from torch.utils.data import DataLoader
from scipy import stats

from egg.zoo.external_game.archs import Sender, Receiver, ReinforceReceiver


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the unizped VisA dataset')
    parser.add_argument('--valid_prop', type=float, default=0.2,
                        help='Proportion of dataset to use for validation')
    parser.add_argument('--n_distractors', type=int, default=4,
                        help='Number of distractors for receiver to see')
    parser.add_argument('--dump_data', type=str, default=None,
                        help='Path to the data for which to produce output information')
    parser.add_argument('--dump_output', type=str, default=None,
                        help='Path for dumping output information')

    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')

    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
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
        core.dump_sender_receiver(game, dataset, gs=is_gs, device=device, variable_length=True)

    for sender_input, message, receiver_output, label \
            in zip(sender_inputs, messages, receiver_outputs, labels):
        sender_input = ' '.join(map(str, sender_input.tolist()))
        message = ' '.join(map(str, message.tolist()))
        if is_gs: receiver_output = receiver_output.argmax()
        print(f'{sender_input};{message};{receiver_output};{label.item()}')


def topographical_similarity(inputs, messages):
    dist = lambda x, y: (x != y).sum(-1)
    dists_x = [dist(x1, x2) for i, x1 in enumerate(inputs) for x2 in inputs[i:]]
    dists_y = [
        dist(y1, y2)
        for i, y1 in enumerate(messages.argmax(-1))
        for y2 in messages[i:].argmax(-1)
    ]
    # spearmanr complains about dividing by a 0 stddev sometimes; just let it nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        corr = stats.spearmanr(dists_x, dists_y)[0]
        if math.isnan(corr):
            corr = 0
        return  corr


def differentiable_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    labels = labels.squeeze(1)
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    toposim = topographical_similarity(_sender_input, _message)
    toposim = torch.FloatTensor([toposim]*len(acc))
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {'acc': acc, 'toposim': toposim}


def non_differentiable_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    labels = labels.squeeze(1)
    acc = (receiver_output == labels).detach().float()
    return -acc, {'acc': acc}


def build_model(opts, train_loader, dump_loader):
    n_features = train_loader.dataset.get_n_features() if train_loader else dump_loader.dataset.get_n_features()

    if opts.n_classes is not None:
        receiver_outputs = opts.n_classes
    else:
        receiver_outputs = train_loader.dataset.get_output_max() + 1 if train_loader else \
                dump_loader.dataset.get_output_max() + 1

    sender = Sender(n_hidden=opts.sender_hidden, n_features=n_features)

    receiver = Receiver(output_size=receiver_outputs, n_hidden=opts.receiver_hidden)
    loss = differentiable_loss
    # TODO: implement non_differentiable_loss for rf?
    # if opts.train_mode.lower() == 'gs':
    #     loss = differentiable_loss
    #     receiver = Receiver(output_size=receiver_outputs, n_hidden=opts.receiver_hidden)
    # else:
    #     loss = non_differentiable_loss
    #     receiver = ReinforceReceiver(output_size=receiver_outputs, n_hidden=opts.receiver_hidden)

    return sender, receiver, loss


if __name__ == "__main__":
    opts = get_params()

    print(f'Launching game with parameters: {opts}')

    device = torch.device("cuda" if opts.cuda else "cpu")

    if opts.data_path is None:
        raise ValueError("--data_path must be supplied")
    whole_dataset = VisaDataset.from_xml_files(opts.data_path, opts.n_distractors)
    validation_dataset, train_dataset = whole_dataset.valid_train_split(opts.valid_prop)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=1
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=1
    )


    dump_loader = None
    if opts.dump_data:
        dump_loader = DataLoader(CSVDataset(path=opts.dump_data),
                                 batch_size=opts.batch_size,
                                 shuffle=False, num_workers=1)

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
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=validation_loader, callbacks = [core.ConsoleLogger(print_train_loss=False, print_test_loss=True)])

    if dump_loader is not None:
        if opts.dump_output:
            with open(opts.dump_output, 'w') as f, contextlib.redirect_stdout(f):
                dump(game, dump_loader, device, opts.train_mode.lower() == 'gs')
        else:
            dump(game, dump_loader, device, opts.train_mode.lower() == 'gs')
    else:
        trainer.train(n_epochs=opts.n_epochs)
        if opts.checkpoint_dir:
            trainer.save_checkpoint()

    core.close()

