from typing import Tuple

from torch.utils.data import Dataset
from torch import from_numpy, LongTensor, Tensor
import pandas as pd
import numpy as np


class ShenParcelDataset(Dataset):
    def __init__(self, shen_file: str, labels_file: str, label_col: str, max_seq_length: int = -1, min_seq_length: int = 1, total_time = 15 * 60, seperator='\t'):
        super().__init__()
        self.scans = pd.read_csv(shen_file, sep=seperator)
        self.labels = pd.read_csv(labels_file, sep=seperator)
        self.label_col = label_col
        self.classes = self.labels[label_col].unique()
        self.scans_tr = total_time / len(self.scans)
        self.max_seq_length = max_seq_length
        # remove all labels with too short an exposure
        rows_to_remove = self.labels.duration < min_seq_length * self.scans_tr
        self.labels = self.labels.drop(self.labels.index[rows_to_remove])

    def __len__(self) -> int:
        return len(self.labels[self.label_col])

    def __get__(self, idx: int) -> Tuple[Tensor, LongTensor]:
        exposure = self.labels.iloc[idx]
        label = self.classes.index(exposure[self.label_col])
        onset = exposure.onset
        duration = exposure.duration
        start_tr = int(onset) // self.scans_tr
        end_tr = int(onset + duration) // self.scans_tr

        # if the specific exposure is longer than the max length we wish to train on
        if (self.max_seq_length > -1) and ((end_tr - start_tr) > self.max_seq_length):
            rand_sample = np.random.uniform()
            # We randomly sample a start TR that could be used as the start of the maximum length possible
            start_tr = int(((end_tr - start_tr) - self.max_seq_length) * rand_sample)
            end_tr = start_tr + self.max_seq_length

        labelled_scans = self.scans.iloc[range(start_tr, end_tr)]
        tensor_scans = from_numpy(labelled_scans.to_numpy())
        return tensor_scans, LongTensor([label])
