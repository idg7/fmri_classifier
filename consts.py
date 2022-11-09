DEBUG = False
MAX_GRAD_NORM = 0.5
SHEN_PARCEL_DIM = 512
NUM_WORKERS = 4
MIN_SEQ_LENGTH = 1
SCANS_TR = 2

PARCELS_ROOT_PATH = '/galitylab/data/studyforrest-data/derivative/aggregate_fmri_timeseries'
LABELS_ROOT_PATH = '/galitylab/data/studyforrest-data-annotations/segments/avmovie'

MLFLOW_TRACKING_URI = '/galitylab/experiments/mlflow/store'#'http://127.0.0.1:5000' #5000
MLFLOW_ARTIFACT_STORE = '/galitylab/experiments/mlflow/artifact_store' #'/home/hdd_storage/mlflow/artifact_store'
EXPERIMENT_NAME = 'fMRI start-end classifier (forrest)'