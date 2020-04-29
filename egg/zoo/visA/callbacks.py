import math
import warnings
import json
import logging
from typing import Dict, Any, Callable, Optional, List, Union, cast

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
        super(ToposimCallback, self).__init__()
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
        messages_tensor = torch.cat(messages, 0)

        counts = np.unique(messages_tensor, return_counts=True)[1]
        counts = np.array(sorted(counts, key=lambda x: -x), dtype=np.float32)
        word_freqs = counts / counts.sum()

        if self.use_embeddings:
            embeddings = self.sender.embedding.weight.transpose(0, 1).detach()
            toposim = calculate_toposim(
                torch.cat(inputs, 0),
                embeddings[messages_tensor],
                cosine_dist,
                lambda x: cosine_dist(x, reduce_dims=(-2, -1)),
            )
        else:
            toposim = calculate_toposim(
                torch.cat(inputs, 0), messages_tensor, cosine_dist, levenshtein_dist
            )
        if logs is not None:
            logs["word_freqs"] = word_freqs
            logs["toposim"] = toposim


class MetricLogger(Callback):
    def __init__(self) -> None:
        super(MetricLogger, self).__init__()
        self._finalized_logs: Optional[Dict[str, Any]] = None
        self._train_logs: List[Dict[str, Any]] = []
        self._valid_logs: List[Dict[str, Any]] = []

    @staticmethod
    def _detach_tensors(d: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.detach().numpy() if torch.is_tensor(v) else v for k, v in d.items()
        }

    def on_test_end(self, loss: float, logs: Dict[str, Any]) -> None:
        log_dict = MetricLogger._detach_tensors({"loss": loss, **logs})
        self._valid_logs.append(log_dict)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any]) -> None:
        log_dict = MetricLogger._detach_tensors({"loss": loss, **logs})
        self._train_logs.append(log_dict)

    @staticmethod
    def _dicts_to_arrays(dict_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        train_lists: Dict[str, List] = {}
        for d in dict_list:
            for field, value in d.items():
                if field not in train_lists:
                    train_lists[field] = []
                if torch.is_tensor(value):
                    value = value.detach().numpy()
                train_lists[field].append(value)
        return {k: np.array(v) for k, v in train_lists.items()}

    def on_train_end(self) -> None:
        assert len(self._train_logs) > 0
        assert len(self._valid_logs) > 0
        train_logs = MetricLogger._dicts_to_arrays(self._train_logs)
        valid_logs = MetricLogger._dicts_to_arrays(self._valid_logs)
        # TODO Add other post-processing of metrics
        self._finalized_logs = {"train": train_logs, "valid": valid_logs}

    def get_finalized_logs(self) -> Dict[str, Any]:
        if self._finalized_logs is None:
            raise ValueError("Logs are not yet finalized.")
        else:
            return self._finalized_logs


class ConsoleLogger(Callback):
    def __init__(self, print_train_loss=False, as_json=False, print_test_loss=True):
        self.print_train_loss = print_train_loss
        self.as_json = as_json
        self.epoch_counter = 0
        self.print_test_loss = print_test_loss

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None):
        if logs is None:
            logs = {}
        if self.print_test_loss:
            if self.as_json:
                dump = dict(
                    mode="test", epoch=self.epoch_counter, loss=self._get_metric(loss)
                )
                for k, v in logs.items():
                    dump[k] = self._get_metric(v)
                output_message = json.dumps(dump)
            else:
                output_message = (
                    f"test: epoch {self.epoch_counter}, loss {loss:.4f},  {logs}"
                )
            logging.info(output_message)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        if logs is None:
            logs = {}
        self.epoch_counter += 1

        if self.print_train_loss:
            if self.as_json:
                dump = dict(
                    mode="train", epoch=self.epoch_counter, loss=self._get_metric(loss)
                )
                for k, v in logs.items():
                    dump[k] = self._get_metric(v)
                output_message = json.dumps(dump)
            else:
                output_message = (
                    f"train: epoch {self.epoch_counter}, loss {loss:.4f},  {logs}"
                )
            logging.info(output_message)

    def _get_metric(self, metric: Union[torch.Tensor, float]) -> float:
        if torch.is_tensor(metric) and cast(torch.Tensor, metric).dim() > 1:
            return cast(torch.Tensor, metric).mean().item()
        elif torch.is_tensor(metric):
            return cast(torch.Tensor, metric).item()
        elif type(metric) == float:
            return metric
        else:
            raise TypeError("Metric must be either float or torch.Tensor")


class VocabCountsReset(Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_begin(self):
        if hasattr(self.model, "reset_counts"):
            self.model.reset_counts()
