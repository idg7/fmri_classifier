from .kfold_dataset import KFoldDataset
from .multi_segment_shen_dataset import MultiSegmentShenParcelDataset
from .multi_fold_dataset import MultiFoldDataset
from typing import List, Tuple, Dict
import numpy as np
from torch.utils.data import DataLoader
from consts import NUM_WORKERS
from .utils import collate_fn_diff_size_scans, collate_fn_not_seq


def get_kfolds(k: int, batch_size: int, 
                shen_files: List[str], subj_idx: List[int], 
                labels_files: List[str], label_map: Dict[str, int], 
                label_col: str, max_seq_length: int = -1, use_mlp: bool = False) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, List[int]]:
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
    collate_fn = collate_fn_diff_size_scans
    if use_mlp:
        max_seq_length = 1
        collate_fn = collate_fn_not_seq

    entire_ds = MultiSegmentShenParcelDataset(shen_files, subj_idx, labels_files, label_map, label_col, max_seq_length=max_seq_length)
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
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS)
        for set in training_sets]
    
    validation_sets = [
        DataLoader(
            set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS)
        for set in folds]
    
    entire_set = DataLoader(
            entire_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS)

    return training_sets, validation_sets, entire_set, entire_ds.get_parcelation_sizes()
