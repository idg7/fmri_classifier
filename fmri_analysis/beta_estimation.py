from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from .roi import get_roi
from data import merge_consecutive_labels
from .consts import *
from .create_predictors_mat import *



def get_character_betas_all(rois: List[np.ndarray], labels: pd.DataFrame, character: str, cross_validated: bool = True, normalize_trs: bool = True, predict_characters_level: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
    cross_val_betas = []
    num_TRs = rois[0].shape[1]
    
    # We get the label vector
    if predict_characters_level:
        y = create_character_level_label_vector(labels, character, num_TRs)
    else:
        y = create_label_vector(labels, character, num_TRs)

    idx_A = None
    for roi in rois:
        all = np.concatenate((roi, y)).T
        ds = pd.DataFrame(all, columns = [f'X{i}' for i in range(roi.shape[0])] + ['y'], index=[i for i in range(num_TRs)])

        # Filter just labels s.t. y in {0, 1}
        ds = ds.loc[ds['y'] >= 0, :]
        
        # If we haven't split the ds to 50/50 portions for cross validation, do it now
        if idx_A is None:
            ds_A = ds.sample(frac=0.5)
            idx_A = ds_A.index
        
        ds_A = ds.loc[idx_A]
        ds_B = ds.drop(idx_A)

        # Normalize all voxels to get the beta coefficients
        # With TR normalization
        if normalize_trs:
            normalized_vox_A = stats.zscore(stats.zscore(ds_A.drop('y', axis=1), axis=0), axis=1)
            normalized_vox_B = stats.zscore(stats.zscore(ds_B.drop('y', axis=1), axis=0), axis=1)

        # Without TR normalization
        else:
            normalized_vox_A = stats.zscore(ds_A.drop('y', axis=1), axis=1)
            normalized_vox_B = stats.zscore(ds_B.drop('y', axis=1), axis=1)

        # Set linear models, and log the beta coefficient
        model_A = sm.OLS(ds_A['y'], normalized_vox_A)
        model_B = sm.OLS(ds_B['y'], normalized_vox_B)
        cross_val_betas.append((model_A.fit().params, model_B.fit().params))

    return cross_val_betas


def get_all_subjs_segment_betas(segment: int, region: str, normalize_trs: bool = True, predict_characters_level: bool = False) -> Tuple[List[str], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    We return:
        List of valid characters in the segment,
        Dictionary matching:
            the subject index,
            Tuple of:
                betas matrix of partition A,
                betas matrix of partition B
    """
    subj_betas = {subj:[] for subj in VALID_SUBJ_IDS}
    valid_chars = []
    rois = []

    # Load the roi for each subject
    for subj in VALID_SUBJ_IDS:
        rois.append(get_roi(subj, segment, region))

    # Load the labels
    labels = pd.read_csv(f'/galitylab/data/studyforrest-data-annotations/segments/avmovie/emotions_av_1s_events_run-{segment}_events.tsv', sep='\t')
    labels = merge_consecutive_labels(labels, LABEL_COL)
    
    for character in ['FORREST', 'JENNY', 'DAN', 'BUBBA', 'MRSGUMP']:
        if labels[LABEL_COL].str.contains(character).sum() > 0:
            # for each valid character of interest, calculate it's betas
            valid_chars.append(character)
            char_betas = get_character_betas_all(rois, labels, character, normalize_trs=normalize_trs, predict_characters_level=predict_characters_level)
            
            # Reorder betas so all vectors belonging to specific subject are the same
            for i in range(len(char_betas)):
                subj_betas[VALID_SUBJ_IDS[i]].append(char_betas[i])

    # Concatenate the betas
    for subj in VALID_SUBJ_IDS:
        A = [np.expand_dims(subj_betas[subj][j][0],0) for j, _ in enumerate(valid_chars)]
        B = [np.expand_dims(subj_betas[subj][j][1],0) for j, _ in enumerate(valid_chars)]
        subj_betas[subj] = (np.concatenate(A,0), np.concatenate(B,0))

    return valid_chars, subj_betas
