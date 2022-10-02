import torch
from kfold_dataset import KFoldDataset
from multi_segment_shen_dataset import MultiSegmentShenParcelDataset
from typing import List
import numpy as np
from torch.utils.data import DataLoader


def collate_fn(batch):
    scans_batch, label_batch = list(zip(*batch))
    min_length = min(scans.shape[0] for scans in scans_batch if len(scans)>0)
    scans_batch = [scans[:min_length] for scans in scans_batch if len(scans)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, scans_batch) if len(imgs)>0]
    scans_tensor = torch.stack(scans_batch)
    labels_tensor = torch.stack(label_batch)
    return scans_tensor, labels_tensor    


def get_kfolds(k: int, batch_size: int, shen_files: List[str], labels_files: List[str], label_col: str) -> List[KFoldDataset]:
    ds = MultiSegmentShenParcelDataset(shen_files, labels_files, label_col)
    item_indices = np.arange(len(ds))
    chunks_size = len(ds) // k

    chunks = [i * chunks_size for i in range(k - 1)]
    chunks = np.split(np.random.shuffle(item_indices), chunks)
    
    folds = [
        DataLoader(
            KFoldDataset(ds, chunk), 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4)
        for chunk in chunks]
    return folds