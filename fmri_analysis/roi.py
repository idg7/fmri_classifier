import numpy as np
from glob import glob
import nilearn as nl
import nibabel as nib
from nilearn import plotting, image


def get_full_roi_mask(subj: int, region: str) -> np.ndarray:
    subj_query = '/galitylab/data/studyforrest-data/derivative/visual_areas/sub-{subj:02d}/rois/*{region}_*_mask.nii.gz'.format(subj=subj, region=region)
    roi_paths = glob(subj_query)
    full_mask = 0
    for roi_path in roi_paths:
        mask = image.load_img(roi_path)
        mask_data = mask.get_fdata()
        full_mask = full_mask + mask_data
    return full_mask
    

def get_roi(subj: int, segment: int, region: str) -> np.ndarray:
    func_path = '/galitylab/data/studyforrest-data/derivative/aligned_mri/sub-{subj:02d}/in_bold3Tp2/sub-{subj:02d}_task-avmovie_run-{segment}_bold.nii.gz'.format(subj=subj,segment=segment)
    func = image.load_img(func_path)
    func_data = func.get_fdata()

    full_mask = get_full_roi_mask(subj, region)
    
    roi = func_data[full_mask >= 1]
    if np.sum(full_mask >= 1) == 0:
        print(f'filtered ROI shape: {roi.shape}')
        print(f'Num voxels: {np.sum(full_mask >= 1)}')
    return roi