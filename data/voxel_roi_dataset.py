from typing import Tuple, Dict, List
from .utils import merge_consecutive_labels
from torch.utils.data import Dataset
from torch import from_numpy, LongTensor, Tensor
import pandas as pd
import numpy as np


class VoxelROIDataset(Dataset):
    def __init__(self, func_files: List[str], roi_files: List[str]):
        super().__init__()
        roi_mask = get_roi_mask(roi_files)
        roi_mask = filter_func(roi_files)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, int]:
        raise NotImplementedError
        