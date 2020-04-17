import logging
from argparse import Namespace
from typing import Iterator

import torch

from egg.zoo.visA import game

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

def copy(ns: Namespace) -> Namespace:
    """Peroform a shallow copy on the given namespace."""
    return Namespace(**vars(ns))

default_opts = Namespace(
    batch_size=32,
    checkpoint_dir=None,
    checkpoint_freq=0,
    cuda=False,
    data_path="data/visa/US",
    data_set="visa",
    device=torch.device(type="cpu"),
    dump_data=None,
    dump_output=None,
    early_stopping_thr=0.98,
    examples_per_epoch=1000,
    force_eos=False,
    load_from_checkpoint=None,
    lr=0.001,
    max_len=5,
    n_classes=None,
    n_distractors=9,
    n_epochs=4,
    no_cuda=True,
    optimizer="adam",
    preemptable=False,
    random_seed=0,
    receiver_cell="gru",
    receiver_embedding=50,
    receiver_entropy_coeff=0.01,
    receiver_hidden=20,
    receiver_layers=1,
    receiver_lr=0.001,
    sender_cell="gru",
    sender_embedding=50,
    sender_entropy_coeff=0.01,
    sender_hidden=20,
    sender_layers=1,
    sender_lr=0.001,
    temperature=1.0,
    tensorboard=False,
    tensorboard_dir="runs/",
    toposim_embed=True,
    train_mode="gs",
    valid_prop=0.2,
    validation_freq=1,
    vocab_size=10,
)

def opt_generator(base_opts: Namespace) -> Iterator[Namespace]:
    for i in range(1, 4):
        opts = copy(base_opts)
        opts.max_len = i
        yield opts

def main():
    # TODO When we run the game here, we are skipping the "important" intialization
    # of the EGG framework, but this involves editing global state which is horrible
    # for doing things programmatically. If something doesn't get initialized, this is
    # probably why.
    for opts in opt_generator(default_opts):
        output = game.run_game(opts)
        print("objective: " + str(output['objective']))


if __name__ == "__main__":
    main()
