from typing import Tuple, List
from torch.utils.data import Dataset
from torch import Tensor, LongTensor


class KFoldDataset(Dataset):
    def __init__(self, inner_ds: Dataset, whitelist: List[int]):
        super().__init__()
        self.inner = inner_ds
        self.whitelist = whitelist
        self.total_labels = self.inner.total_labels

    def __len__(self) -> int:
        return len(self.whitelist)

    def __getitem__(self, idx: int) -> Tuple[Tensor, LongTensor]:
        return self.inner[self.whitelist[idx]]