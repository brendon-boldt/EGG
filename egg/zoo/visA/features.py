# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional
from torch.utils.data import Dataset
import torch
from torch import Tensor, LongTensor
import numpy as np
from scipy.io import loadmat

# TODO: Allow for distractor resampling for training data.


class DistractorDataset(Dataset):
    """Base dataset for providing examples and distractors."""

    def __init__(
        self,
        items: Union[List, np.ndarray, Tensor],
        classes: Optional[Union[List, np.ndarray]] = None,
        resample: bool = False,
        n_distractors=5,
        random_seed=None,
    ) -> None:
        super(DistractorDataset, self).__init__()
        self.resample = resample
        self.n_distractors = n_distractors
        self.items = Tensor(items)
        self.distractor_probs: List[np.ndarray] = []
        self.random_state = np.random.RandomState(random_seed)
        if classes is None:
            # If we are not using classes, treat each item as its own class
            self.classes = np.arange(self.items.shape[0])
        else:
            self.classes = np.array(classes)

        def exclude_p(n: int, idxs: Union[List[int], np.ndarray]) -> np.ndarray:
            p = np.full(n, 1 / (n - len(idxs)))
            p[idxs] = 0.0
            return p

        assert self.classes.shape[0] == self.items.shape[0]
        class_idxs: Dict[str, List[int]] = {}
        for i, class_name in enumerate(self.classes):
            if class_name not in class_idxs:
                class_idxs[class_name] = []
            class_idxs[class_name].append(i)
        for i in range(len(self.items)):
            p = exclude_p(len(self.items), class_idxs[self.classes[i]])
            self.distractor_probs.append(p)
        if not self.resample:
            self._populate_distractors()

    def _sample_distractors(self, idx: int) -> Tuple[int, np.ndarray]:
        probs = self.distractor_probs[idx]
        distractor_idxs = self.random_state.choice(
            self.items.shape[0], self.n_distractors + 1, replace=False, p=probs
        )
        true_place = self.random_state.randint(0, self.n_distractors + 1)
        distractor_idxs[true_place] = idx
        return true_place, distractor_idxs

    def _populate_distractors(self) -> None:
        self.distractor_idxs: List[Tuple[int, np.ndarray]] = []
        for i in range(len(self.items)):
            true_idx, distractor_idxs = self._sample_distractors(i)
            self.distractor_idxs.append((true_idx, distractor_idxs))

    def valid_train_split(
        self, valid_prop
    ) -> Tuple["DistractorDataset", "DistractorDataset"]:
        valid_len = int(len(self) * valid_prop)
        idxs = np.random.choice(len(self), len(self), replace=False)
        valid_idxs = idxs[:valid_len]
        train_idxs = idxs[valid_len:]
        # TODO Account for random seed
        valid_ds = self.__class__(
            self.items[valid_idxs],
            classes=self.classes[valid_idxs],
            n_distractors=self.n_distractors,
            resample=False,
            random_seed=None,
        )
        train_ds = self.__class__(
            self.items[train_idxs],
            classes=self.classes[train_idxs],
            n_distractors=self.n_distractors,
            resample=True,
            random_seed=None,
        )
        return valid_ds, train_ds

    def get_n_features(self) -> int:
        return self.items.shape[-1]

    def get_output_size(self) -> int:
        # I do not know what this is supposed to return.
        raise NotImplementedError

    def get_output_max(self) -> int:
        return self.n_distractors

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx) -> Tuple[Tensor, LongTensor, Tensor]:
        if self.resample:
            true_idx, dist_idxs = self._sample_distractors(idx)
        else:
            true_idx, dist_idxs = self.distractor_idxs[idx]
        distractors = torch.stack([self.items[i] for i in dist_idxs], 0)
        return self.items[idx], LongTensor([true_idx]), distractors


class GroupedInaDataset(DistractorDataset):
    @staticmethod
    def from_file(path, n_distractors, random_seed=None):
        raise NotImplementedError
        train_ds = InaDataset()
        test_ds = InaDataset()
        train_ds.n_distractors = n_distractors
        test_ds.n_distractors = n_distractors
        path = Path(path)
        np_file = np.load(path)
        x_train = np_file["x_train"]
        y_train = np_file["y_train"]
        x_test = np_file["x_test"]
        y_test = np_file["y_test"]

        train_ds._build_frames(x_train, random_seed, y_train)
        test_ds._build_frames(x_test, random_seed, y_test)
        return train_ds, test_ds


class InaDataset(DistractorDataset):
    """Dataset for ImageNet Atttributes

    TODO: Insert URL
    """

    @staticmethod
    def from_file(path, n_distractors, random_seed=None):
        """Dataset for loading VisA data from XML format
        Row format (3-tuple):
        - (n_features,) shaped array of dtype bool
        - integer index of the true vector
        - (n_distractors, n_features) shaped array of dtype bool

        """
        raise NotImplementedError

        # TODO add option for same class or same object

        ds = InaDataset()
        ds.n_distractors = n_distractors

        path = Path(path)
        raw_data = loadmat(path)["attrann"][0][0]
        attr_arr = raw_data[2].squeeze().astype(np.float32)
        ids = [raw_data[0][i][0][0] for i in range(len(raw_data[0]))]
        classes = [item_id.split("_")[0] for item_id in ids]
        ds.attr_arr = attr_arr
        ds.classes = classes
        ds._build_frames(attr_arr, random_seed, classes)
        return ds


class VisaDataset(DistractorDataset):
    @staticmethod
    def from_file(path, n_distractors, random_seed=None) -> "VisaDataset":
        """Dataset for loading VisA data from XML format
        Row format (3-tuple):
        - (n_features,) shaped array of dtype bool
        - integer index of the true vector
        - (n_distractors, n_features) shaped array of dtype bool

        """
        rows = []
        path = Path(path)
        for xml_path in path.iterdir():
            rows += VisaDataset._parse_xml_file(xml_path)
        attr_arr = VisaDataset._rows_to_array(rows)
        # ds._build_frames(attr_arr, random_seed)
        ds = VisaDataset(attr_arr, n_distractors=n_distractors, random_seed=random_seed)
        return ds

    @staticmethod
    def _parse_xml_file(path):
        tree = ET.parse(path)
        root = tree.getroot()
        category = root.get("category")
        members = []
        for concept in root.findall(".//concept"):
            attributes = set()
            for group in concept:
                atts = ET.tostring(group, method="text", encoding="unicode")
                attributes |= {att.strip() for att in atts.strip().split("\n")}
            row = (category, concept.get("name"), attributes)
            members.append(row)
        return members

    @staticmethod
    def _rows_to_array(ds):
        attrs_set = set()
        for x in ds:
            attrs_set |= x[2]
        attrs = {a: i for i, a in enumerate(attrs_set)}
        arr = np.zeros((len(ds), len(attrs_set)), dtype=np.float32)
        for i, row in enumerate(ds):
            for attr in row[2]:
                arr[i][attrs[attr]] = 1.0
        return arr
