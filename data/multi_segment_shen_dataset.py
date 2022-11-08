from typing import Tuple, List, Dict

from torch.utils.data import Dataset
from torch import LongTensor, Tensor
import pandas as pd
from data.shen_dataset import ShenParcelDataset


class MultiSegmentShenParcelDataset(Dataset):
    def __init__(self, shen_files: List[str], subj_idx: List[int], labels_files: List[str], label_map: Dict[str, int], label_col: str, segment_total_time: int = 15 * 60, seperator='\t'):
        super().__init__()
        assert(len(shen_files) == len(labels_files))

        self.segments_datasets = [
            ShenParcelDataset(
                shen_files[i],
                labels_files[i],
                label_col,
                subj_idx[i], label_map,
                total_time=segment_total_time,
                seperator=seperator) for i in range(len(shen_files))
        ]

        self.total_size = sum([len(segment) for segment in self.segments_datasets])

        self.all_segments_labels = pd.concat([segment.labels for segment in self.segments_datasets])
        self.total_labels = label_map

        # Get the parcelation dims from every subject
        max_idx = 0
        for segment in self.segments_datasets:
            max_idx = max(max_idx, segment.subj_idx)
        self.parcelation_sizes = [0] * (max_idx + 1)
        
        for segment in self.segments_datasets:
            self.parcelation_sizes[segment.subj_idx] = segment.get_parcelation_size()
    
    def get_parcelation_sizes(self) -> List[int]:
        return self.parcelation_sizes

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, int]:  # type: ignore
        assert(idx < self.total_size)
        for segment in self.segments_datasets:
            if idx >= len(segment):
                idx -= len(segment)
            else:
                return segment[idx]