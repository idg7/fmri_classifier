import nilearn as nl
from typing import List
import numpy as np


def get_roi_mask(rois_paths: List[str]) -> np.ndarray:
    # Sum all relevant ROI masks for the functional data
    rois = [nl.image.load_img(roi_pth) for roi_pth in rois_paths]
    full_mask = np.zeros(rois[0].shape)
    for roi in rois:
        full_mask += roi.get_fdata()
    
    full_mask = (full_mask > 0)
    
    return full_mask


def filter_func(func_paths: List[str], roi_mask: np.ndarray) -> np.ndarray:
    funcs = [nl.image.load_img(func_path).get_fdata() for func_path in func_paths]
    unified = np.concatenate(funcs, axis=1)
    return unified[roi_mask]


if __name__ == '__main__':
    subject_ids = [f'{i:02d}' for i in [1,2,3,4,5,6,9,10,14,15,16,17,18,19,20]]
    subject_func_dirs = [
        f'/galitylab/data/studyforrest-data/derivative/aligned_mri/sub-{subj}]/in_bold3Tp2/sub-{subj}_task-avmovie_run-{run}_bold.nii.gz'
        for subj in subject_ids for run in range(1,9)
    ]


