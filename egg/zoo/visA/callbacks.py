import math
import warnings
from typing import Dict, Any, Callable, Optional, List

import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats

from egg.core import Callback, move_to, Trainer


def cosine_dist(vecs: torch.Tensor, reduce_dims=-1) -> torch.Tensor:
    vecs /= vecs.norm(dim=-1, keepdim=True)
    return -(vecs.unsqueeze(0) * vecs.unsqueeze(1)).sum(reduce_dims)


def levenshtein_dist(msgs: torch.Tensor) -> torch.Tensor:
    return (msgs.unsqueeze(0) != msgs.unsqueeze(1)).sum(-1)


def get_upper_triangle(x: np.ndarray) -> np.ndarray:
    return x[np.triu_indices(x.shape[0])].reshape(-1)


def calculate_toposim(
    inputs: torch.Tensor,
    messages: torch.Tensor,
    input_dist: Callable,
    message_dist: Callable,
) -> float:
    in_dists = get_upper_triangle(input_dist(inputs).cpu().numpy())
    msg_dists = get_upper_triangle(message_dist(messages).cpu().numpy())
    # spearmanr complains about dividing by a 0 stddev sometimes; just let it nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        corr = stats.spearmanr(in_dists, msg_dists)[0]
        if math.isnan(corr):
            corr = 0
        return corr


class ToposimCallback(Callback):
    trainer: "Trainer"

    def __init__(
        self, valid_dl: DataLoader, train_dl: DataLoader, sender, use_embeddings=True
    ) -> None:
        self.use_embeddings = use_embeddings
        self.valid_dl = valid_dl
        self.train_dl = train_dl
        self.sender = sender

    def on_test_end(self, *args, **kwargs) -> None:
        return self._caluclate_toposim(self.valid_dl, *args, **kwargs)

    def on_epoch_end(self, *args, **kwargs) -> None:
        return self._caluclate_toposim(self.train_dl, *args, **kwargs)

    def _caluclate_toposim(
        self, dataloader: DataLoader, loss: float, logs: Dict[str, Any] = None
    ) -> None:
        sender_mode = self.sender.training
        self.sender.eval()
        messages: List[torch.Tensor] = []
        inputs = []
        # Ignore repeats for toposim calculation
        n_repeats = dataloader.dataset.n_repeats
        dataloader.dataset.n_repeats = 1
        with torch.no_grad():
            for batch in dataloader:
                # batch = move_to(batch, self.sender.device)
                inputs.append(batch[0])
                output = self.sender(batch[0])
                # TODO Determine a better way to do this
                # If the method RF, the output is a tuple
                if type(output) == tuple:
                    messages.append(output[0])
                else:
                    messages.append(output.argmax(-1))
        dataloader.dataset.n_repeats = n_repeats
        self.sender.train(sender_mode)
        if self.use_embeddings:
            embeddings = self.sender.embedding.weight.transpose(0, 1).detach()
            toposim = calculate_toposim(
                torch.cat(inputs, 0),
                embeddings[torch.cat(messages, 0)],
                cosine_dist,
                lambda x: cosine_dist(x, reduce_dims=(-2, -1)),
            )
        else:
            toposim = calculate_toposim(
                torch.cat(inputs, 0),
                torch.cat(messages, 0).argmax(-1),
                cosine_dist,
                levenshtein_dist,
            )
        if logs is not None:
            logs["toposim"] = toposim
