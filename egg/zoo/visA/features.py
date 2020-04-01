# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np
from scipy.io import loadmat

# TODO: Allow for distractor resampling for training data.


class DistractorDataset(Dataset):
    """Base dataset for providing examples and distractors."""

    def _build_frames(self, attr_arr, random_seed, classes=None):
        def exclude_p(n, idxs):
            p = np.full(n, 1 / (n - len(idxs)))
            p[idxs] = 0.0
            return p

        if classes is None:
            # If we are not using classes, treat each item as its own class
            classes = list(range(attr_arr.shape[0]))

        assert len(classes) == attr_arr.shape[0]
        class_idxs = {}
        for i, class_name in enumerate(classes):
            if class_name not in class_idxs:
                class_idxs[class_name] = []
            class_idxs[class_name].append(i)

        r = np.random.RandomState(random_seed)
        self.frames = []
        for i in range(len(attr_arr)):
            p = exclude_p(len(attr_arr), class_idxs[classes[i]])
            distractor_idxs = r.choice(
                len(attr_arr), self.n_distractors + 1, replace=False, p=p
            )
            distractors = attr_arr[distractor_idxs]
            true_idx = r.randint(0, self.n_distractors + 1)
            distractors[true_idx] = attr_arr[i]
            frame = (
                torch.Tensor(attr_arr[i]),
                np.array([true_idx]),
                torch.Tensor(distractors),
            )
            self.frames.append(frame)

    @classmethod
    def from_frames(klass, frames):
        ds = klass()
        ds.frames = frames
        ds.n_distractors = frames[0][2].shape[0] - 1
        return ds

    def valid_train_split(self, valid_prop):
        valid_len = int(len(self) * valid_prop)
        idxs = np.random.choice(len(self), len(self), replace=False)
        valid_frames = []
        train_frames = []
        for i, idx in enumerate(idxs):
            if i < valid_len:
                valid_frames.append(self[idx])
            else:
                train_frames.append(self[idx])
        return (
            self.__class__.from_frames(valid_frames),
            self.__class__.from_frames(train_frames),
        )

    def get_n_features(self):
        return self.frames[0][0].shape[0]

    def get_output_size(self):
        # I do not know what this is supposed to return.
        raise NotImplementedError

    def get_output_max(self):
        # return max(x[1].item() for x in self.frame)
        return self.n_distractors

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


class GroupedInaDataset(DistractorDataset):
    @staticmethod
    def from_file(path, n_distractors, random_seed=None):
        train_ds = InaDataset()
        test_ds = InaDataset()
        train_ds.n_distractors = n_distractors
        test_ds.n_distractors = n_distractors
        path = Path(path)
        np_file = np.load(path)
        x_train = np_file['x_train']
        y_train = np_file['y_train']
        x_test = np_file['x_test']
        y_test = np_file['y_test']

        train_ds._build_frames(x_train, random_seed, y_train)
        test_ds._build_frames(x_test, random_seed, y_test)
        return train_ds, test_ds



class InaDataset(DistractorDataset):
    """Dataset for ImageNet Atttributes

    TODO: Insert URL
    """

    @staticmethod
    def from_mat_file(path, n_distractors, random_seed=None):
        """Dataset for loading VisA data from XML format
        Row format (3-tuple):
        - (n_features,) shaped array of dtype bool
        - integer index of the true vector
        - (n_distractors, n_features) shaped array of dtype bool

        """

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
    def from_xml_files(path, n_distractors, random_seed=None):
        """Dataset for loading VisA data from XML format
        Row format (3-tuple):
        - (n_features,) shaped array of dtype bool
        - integer index of the true vector
        - (n_distractors, n_features) shaped array of dtype bool

        """
        ds = VisaDataset()
        ds.n_distractors = n_distractors

        rows = []
        path = Path(path)
        for xml_path in path.iterdir():
            rows += ds._parse_xml_file(xml_path)
        attr_arr = ds._rows_to_array(rows)
        ds._build_frames(attr_arr, random_seed)
        return ds

    def _parse_xml_file(self, path):
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

    def _rows_to_array(self, ds):
        attrs_set = set()
        for x in ds:
            attrs_set |= x[2]
        attrs = {a: i for i, a in enumerate(attrs_set)}
        arr = np.zeros((len(ds), len(attrs_set)), dtype=np.float32)
        for i, row in enumerate(ds):
            for attr in row[2]:
                arr[i][attrs[attr]] = 1.0
        return arr
