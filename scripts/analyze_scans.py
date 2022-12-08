import nilearn as nil
from nilearn import plotting
import nibabel as nib


if __name__ == '__main__':
    plotting.plot_img('/galitylab/data/studyforrest-data/derivative/visual_areas/sub-01/rois/rFFA_2_mask.nii.gz').show()