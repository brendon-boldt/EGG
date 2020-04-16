import pathlib

from argparse import Namespace
from typing import Optional, List

import torch
from torch.utils.data import DataLoader

from egg.core.util import move_to
from egg.core.trainers import Trainer as CoreTrainer
from egg.core.callbacks import Callback, ConsoleLogger, Checkpoint, CheckpointSaver, TensorboardLogger


class Trainer(CoreTrainer):
    """Clone of the core trainer without depending on global state."""
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: DataLoader,
            opts: Namespace,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ) -> None:
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.train_data = train_data
        self.validation_data = validation_data
        self.validation_freq = opts.validation_freq
        self.device = opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer.state = move_to(self.optimizer.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks

        if opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {opts.load_from_checkpoint}")
            self.load_from_checkpoint(opts.load_from_checkpoint)

        if opts.preemptable:
            assert opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
        else:
            self.checkpoint_path = None if opts.checkpoint_dir is None \
                else pathlib.Path(opts.checkpoint_dir)

        if self.checkpoint_path:
            checkpointer = CheckpointSaver(checkpoint_path=self.checkpoint_path, checkpoint_freq=opts.checkpoint_freq)
            self.callbacks.append(checkpointer)

        if opts.tensorboard:
            assert opts.tensorboard_dir, 'tensorboard directory has to be specified'
            tensorboard_logger = TensorboardLogger()
            self.callbacks.append(tensorboard_logger)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

