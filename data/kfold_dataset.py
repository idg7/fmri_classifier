from typing import Tuple, List
from torch.utils.data import Dataset


class KFoldDataset(Dataset):
    def __init__(self, inner_ds: Dataset, whitelist: List[int]):
        super().__init__()
        self.inner = inner_ds
        self.whitelist = whitelist

    def __len__(self) -> int:
        return len(self.whitelist)

    def __get__(self, idx: int) -> Tuple:
        return self.inner[self.whitelist[idx]]