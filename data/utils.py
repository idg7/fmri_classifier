import torch
import pandas as pd


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


def collate_fn_not_seq(batch):
    scans_batch, subj_id_batch, label_batch = list(zip(*batch))
    scans_batch = [scans for scans in scans_batch if len(scans)>0]
    label_batch = [l for l, imgs in zip(label_batch, scans_batch) if len(imgs)>0]
    subj_id_batch = [l for l, imgs in zip(subj_id_batch, scans_batch) if len(imgs)>0]
    scans_tensor = scans_batch
    labels_tensor = torch.tensor(label_batch)
    subj_id_tensor = torch.tensor(subj_id_batch)
    return scans_tensor, subj_id_tensor, labels_tensor    


def merge_consecutive_labels(labels: pd.DataFrame, labels_col: str) -> pd.DataFrame:
    merged = {labels_col: [], 'onset': [], 'duration': []}
    for i in range(len(labels)):
        character = labels.iloc[i][labels_col]
        onset = labels.iloc[i].onset
        duration = labels.iloc[i].duration

        if (len(merged[labels_col]) > 0) \
            and (merged[labels_col][-1] == character) \
            and (merged['onset'][-1] + merged['duration'][-1] == labels.iloc[i]['onset']):

            # If the next seen character is the same as the previous one
            # and their both consecutive (onset[i-1] + duration[i-1] == onset[i])
            # we merge the 2 exposures into one
            merged['duration'][-1] += labels.iloc[i]['duration']
        else:
            # Otherwise we add a new exposure
            merged[labels_col].append(character)
            merged['onset'].append(onset)
            merged['duration'].append(duration)
    
    return pd.DataFrame(merged)


