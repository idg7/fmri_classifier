from typing import Tuple, List

from torch.utils.data import Dataset
from torch import LongTensor, Tensor
import pandas as pd
from data.shen_dataset import ShenParcelDataset


class MultiSegmentShenParcelDataset(Dataset):
    def __init__(self, shen_files: List[str], labels_files: List[str], label_col: str, segment_total_time: int = 15 * 60, seperator='\t'):
        super().__init__()
        assert(len(shen_files) == len(labels_files))

        self.segments_datasets = [
            ShenParcelDataset(
                shen_files[i],
                labels_files[i],
                label_col,
                segment_total_time,
                seperator) for i in range(len(shen_files))
        ]

        self.total_size = sum([len(segment) for segment in self.segments_datasets])

        self.all_segments_labels = pd.concat([segment.labels for segment in self.segments_datasets])
        self.total_labels = self.all_segments_labels.unique(label_col)

    def __len__(self) -> int:
        return self.total_size

    def __get__(self, idx: int) -> Tuple[Tensor, LongTensor]:
        for segment in self.segments_datasets:
            if idx >= len(segment):
                idx -= len(segment)
            else:
                return segment[idx]