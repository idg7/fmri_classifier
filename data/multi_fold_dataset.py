from typing import Tuple, overload
from torch.utils.data import Dataset
from torch import Tensor, LongTensor


class MultiFoldDataset(Dataset):
    def __init__(self, inner_ds: Tuple[Dataset]):
        super().__init__()
        self.inner_ds = inner_ds

        self.idx_map = {}
        offset = 0
        for i, ds in enumerate(self.inner_ds):
            self.idx_map = self.idx_map | {k + offset: i for k in range(len(ds))}
            offset += len(ds)
        self.total_labels = self.inner_ds[0].total_labels

    def __len__(self) -> int:
        return len(self.idx_map)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, LongTensor, LongTensor]:
        fold = self.idx_map[idx]
        return self.inner_ds[fold][idx]
