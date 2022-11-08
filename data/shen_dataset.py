from typing import Tuple, Dict
from .utils import merge_consecutive_labels
from torch.utils.data import Dataset
from torch import from_numpy, LongTensor, Tensor
import pandas as pd
import numpy as np


class ShenParcelDataset(Dataset):
    def __init__(self, shen_file: str, labels_file: str, label_col: str, subj_idx: int, label_map: Dict[str, int], max_seq_length: int = -1, min_seq_length: int = 1, total_time: int = 15 * 60, seperator='\t'):
        super().__init__()
        self.shen_file = shen_file
        self.labels_file = labels_file
        self.scans = pd.read_csv(shen_file)
        self.labels = merge_consecutive_labels(pd.read_csv(labels_file, sep=seperator), label_col)
        self.label_col = label_col
        self.classes = self.labels[label_col].unique()
        self.scans_tr = 2 # 2S repetition time (TR) as described in https://www.nature.com/articles/sdata201692
        self.max_seq_length = max_seq_length
        # remove all labels with too short an exposure
        rows_to_remove = self.labels.duration < (min_seq_length * self.scans_tr)
        self.labels = self.labels.drop(self.labels.index[rows_to_remove])
        self.label_map = label_map
        self.labels = self.labels[self.labels[label_col].apply(lambda x: x in self.label_map)]
        self.subj_idx = subj_idx

    def get_parcelation_size(self) -> int:
        return len(self.scans.columns)

    def __len__(self) -> int:
        return len(self.labels[self.label_col])

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, int]:
        """
        Given a sample idx
        Returns a Tensor of the fMRI scan, the index to the specific subject IDx, the label for the specfic scan
        """
        exposure = self.labels.iloc[idx]
        if exposure[self.label_col] not in self.label_map:
            print(exposure)
        label = self.label_map[exposure[self.label_col]]
        onset = exposure.onset
        duration = exposure.duration
        start_tr = int(int(onset) // self.scans_tr)
        end_tr = int(int(onset + duration) // self.scans_tr)

        # if the specific exposure is longer than the max length we wish to train on
        if (self.max_seq_length > -1) and ((end_tr - start_tr) > self.max_seq_length):
            rand_sample = np.random.uniform()
            # We randomly sample a start TR that could be used as the start of the maximum length possible
            start_tr = int(((end_tr - start_tr) - self.max_seq_length) * rand_sample)
            end_tr = int(start_tr + self.max_seq_length)
        
        if len(self.scans) < end_tr or start_tr > len(self.scans) or start_tr < 0:
            print(start_tr, len(self.scans), end_tr)
            print(self.shen_file)
            print(self.labels_file)
        labelled_scans = self.scans.iloc[[i for i in range(start_tr, end_tr)]]
        tensor_scans = from_numpy(labelled_scans.to_numpy())

        return tensor_scans, self.subj_idx, label

