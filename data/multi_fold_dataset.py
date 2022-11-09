from typing import Tuple, overload
from torch.utils.data import Dataset
from torch import Tensor, LongTensor


class MultiFoldDataset(Dataset):
    def __init__(self, inner_ds: Tuple[Dataset]):
        super().__init__()
        self.inner_ds = inner_ds

        self.idx_to_fold = {}       # Given a global index, get the specific fold dataset
        self.idx_to_inner_idx = {}  # Given a globel index, get the fold's inner index
        offset = 0
        for i, ds in enumerate(self.inner_ds):
            self.idx_to_fold = self.idx_to_fold | {k + offset: i for k in range(len(ds))}
            self.idx_to_inner_idx = self.idx_to_inner_idx | {k + offset: k for k in range(len(ds))}
            offset += len(ds)
        self.total_labels = self.inner_ds[0].total_labels

    def __len__(self) -> int:
        return len(self.idx_to_fold)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, LongTensor, LongTensor]:
        fold = self.idx_to_fold[idx]
        inner_idx = self.idx_to_inner_idx[idx]
        return self.inner_ds[fold][inner_idx]
