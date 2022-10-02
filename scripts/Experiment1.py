from argparse import ArgumentParser
from models import TransformerDecoderClassifier
from training import Coach
from data import get_kfolds
from typing import List, Tuple
from os import path


def get_args():
    parser = ArgumentParser()
    parser.add_argument('embedding_dim', type=int, default=273)

    return parser.parse_args()


def get_parcels_labels(segments: List[int]) -> Tuple[List[str], List[str]]:
    parcels_root_path = '/home/hdd_storage/forrest/studyforrest-data/derivative/aggregate_fmri_timeseries'
    labels_root_path = '/home/hdd_storage/forrest/studyforrest-data-annotations/segments/avmovie'
    parcels = []
    labels = []
    for sub in ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']:
        for segment in segments:
            parcels.append(path.join(parcels_root_path, f'sub-{sub}', 'shen_fconn', f'sub-{sub}_task-avmovie_run-{segment}_bold.csv'))
            labels.append(path.join(labels_root_path, f'conf_audio_run-{segment}_events.tsv'))
    return parcels, labels


if __name__ == '__main__':
    args = get_args()
    parcels, labels = get_parcels_labels([1,2,3,4])
    model = TransformerDecoderClassifier()