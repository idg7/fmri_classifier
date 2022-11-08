import torch
from .kfold_dataset import KFoldDataset
from .multi_segment_shen_dataset import MultiSegmentShenParcelDataset
from .multi_fold_dataset import MultiFoldDataset
from typing import List, Tuple, Dict
import numpy as np
from torch.utils.data import DataLoader
from consts import NUM_WORKERS


def collate_fn(batch):
    scans_batch, subj_id_batch, label_batch = list(zip(*batch))
    
    print(subj_id_batch)
    print(label_batch)
    min_length = min(scans.shape[0] for scans in scans_batch if len(scans)>0)
    scans_batch = [scans[:min_length] for scans in scans_batch if len(scans)>0]
    label_batch = [l for l, imgs in zip(label_batch, scans_batch) if len(imgs)>0]
    subj_id_batch = [l for l, imgs in zip(subj_id_batch, scans_batch) if len(imgs)>0]
    
    print(subj_id_batch)
    print(label_batch)
    scans_tensor = torch.stack(scans_batch)
    labels_tensor = torch.tensor(label_batch)
    subj_id_tensor = torch.tensor(subj_id_batch)
    print(subj_id_tensor)
    print(labels_tensor)
    return scans_tensor, subj_id_tensor, labels_tensor    


def collate_fn_diff_size_scans(batch):
    scans_batch, subj_id_batch, label_batch = list(zip(*batch))
    min_length = min(scans.shape[0] for scans in scans_batch if len(scans)>0)
    scans_batch = [scans[:min_length] for scans in scans_batch if len(scans)>0]
    label_batch = [l for l, imgs in zip(label_batch, scans_batch) if len(imgs)>0]
    subj_id_batch = [l for l, imgs in zip(subj_id_batch, scans_batch) if len(imgs)>0]
    scans_tensor = scans_batch
    labels_tensor = torch.tensor(label_batch)
    subj_id_tensor = torch.tensor(subj_id_batch)
    return scans_tensor, subj_id_tensor, labels_tensor    


def get_kfolds(k: int, batch_size: int, shen_files: List[str], subj_idx: List[int], labels_files: List[str], label_map: Dict[str, int], label_col: str) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, List[int]]:
    """
    Get K fold division of a dataset
    :param k: The number of folds to split the dataset to
    :param batch_size: Size of each batch
    :param shen_files: File paths containing the shen parcelations
    :param subj_idx: List of subject IDX per shen file
    :param labels_files: File paths containing the labels for each TR
    :param label_col: The column containing the actual label
    :return: (List of training set dataloaders, List of corresponding validation set dataloaders, entire dataset, list of sizes shen parcelation sizes)
    """
    entire_ds = MultiSegmentShenParcelDataset(shen_files, subj_idx, labels_files, label_map, label_col)
    item_indices = np.arange(len(entire_ds))
    chunks_size = len(entire_ds) // k

    chunks = [i * chunks_size for i in range(1, k)]
    np.random.shuffle(item_indices)
    chunks = np.split(item_indices, chunks)

    folds = [KFoldDataset(entire_ds, chunk) for chunk in chunks]

    training_sets = []
    for val_set in folds:
        training_sets.append(MultiFoldDataset(tuple([fold for fold in folds if fold != val_set])))

    training_sets = [
        DataLoader(
            set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_diff_size_scans,
            num_workers=NUM_WORKERS)
        for set in training_sets]
    
    validation_sets = [
        DataLoader(
            set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_diff_size_scans,
            num_workers=NUM_WORKERS)
        for set in folds]
    
    entire_set = DataLoader(
            entire_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_diff_size_scans,
            num_workers=NUM_WORKERS)

    return training_sets, validation_sets, entire_set, entire_ds.get_parcelation_sizes()
