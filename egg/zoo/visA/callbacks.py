import math
import warnings
from typing import Dict, Any, Callable, Optional

import torch
import numpy as np
from scipy import stats

from egg.core import Callback, move_to, Trainer


# TODO Incorporate embeddings into toposim


def cosine_dist(vecs: torch.Tensor) -> torch.Tensor:
    vecs /= vecs.norm(dim=-1, keepdim=True)
    return -(vecs.unsqueeze(0) * vecs.unsqueeze(1)).sum(-1)


def levenshtein_dist(msgs: torch.Tensor) -> torch.Tensor:
    msgs = msgs.argmax(-1)
    return (msgs.unsqueeze(0) != msgs.unsqueeze(1)).sum(-1)


def get_upper_triangle(x: np.ndarray) -> np.ndarray:
    return x[np.triu_indices(x.shape[0])].reshape(-1)


def topographical_similarity(
    inputs: torch.Tensor,
    messages: torch.Tensor,
    input_dist: Optional[Callable] = None,
    message_dist: Optional[Callable] = None,
) -> float:
    if input_dist is None:
        input_dist = cosine_dist
    if message_dist is None:
        message_dist = levenshtein_dist

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

    def __init__(self, dataset, sender, epochs=100) -> None:
        self.dataset = dataset
        self.sender = sender
        self.epochs = epochs

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None) -> None:
        sender_mode = self.sender.training
        self.sender.eval()
        messages = []
        inputs = []
        with torch.no_grad():
            for batch in self.dataset:
                # batch = move_to(batch, self.sender.device)
                # optimized_loss, rest = self.game(*batch)
                inputs.append(batch[0])
                messages.append(self.sender(batch[0]))
        self.sender.train(sender_mode)
        toposim = topographical_similarity(torch.cat(inputs, 0), torch.cat(messages, 0))
        if logs is not None:
            logs["toposim"] = toposim

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        pass
