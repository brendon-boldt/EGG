# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np


class VisaDataset(Dataset):
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
        sender_inputs = ds._rows_to_array(rows)

        def exclude_p(n, i):
            p = np.full(n, 1 / (n-1))
            p[i] = 0.
            return p

        r = np.random.RandomState(random_seed)
        ds.frames = []
        for i in range(len(sender_inputs)):
            p = exclude_p(len(sender_inputs), i)
            distractor_idxs = r.choice(len(sender_inputs), 5, replace=False, p=p)
            distractors = sender_inputs[distractor_idxs]
            true_idx = r.randint(0, ds.n_distractors + 1)
            distractors[true_idx] = sender_inputs[i]
            frame = (
                torch.Tensor(sender_inputs[i]),
                np.array([true_idx]),
                torch.Tensor(distractors)
            )
            ds.frames.append(frame)
        return ds

    @staticmethod
    def from_frames(frames):
        ds = VisaDataset()
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
            VisaDataset.from_frames(valid_frames),
            VisaDataset.from_frames(train_frames),
        )

    def get_n_features(self):
        return self.frames[0][0].shape[0]

    def get_output_size(self):
        # I do not know what this is supposed to return.
        raise NotImplementedError

    def get_output_max(self):
        # return max(x[1].item() for x in self.frame)
        return self.n_distractors + 1

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

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
