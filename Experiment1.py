from argparse import ArgumentParser
from models import TransformerDecoderClassifier, IndividualModelDecoder
from training import Coach
from typing import List, Tuple, Dict
from os import path
from consts import SHEN_PARCEL_DIM, MLFLOW_ARTIFACT_STORE, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MIN_SEQ_LENGTH, SCANS_TR
from data import get_kfolds
from torch import optim, nn
import numpy as np
import pandas as pd
import mlflow


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--kfolds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_attn_heads', type=int, default=8)
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--lr_reduce_step_size', type=int, default=80)
    parser.add_argument('--val_freq', type=int, default=10)
    parser.add_argument('--max_epoch_len', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=120)
    parser.add_argument('--rnn_num_layers', type=int, default=8)

    return parser.parse_args()


def get_parcels_labels(segments: List[int]) -> Tuple[List[str], List[str], str, List[int], Dict[str, int]]:
    """
    Given a list of segment indices, return the segments for the first experiment
    :param segments: A list of segment indices
    :return: (List of Shen parcelations file paths, List of labels file paths, Label column)
    """
    parcels_root_path = '/home/hdd_storage/forrest/studyforrest-data/derivative/aggregate_fmri_timeseries'
    labels_root_path = '/home/hdd_storage/forrest/studyforrest-data-annotations/segments/avmovie'
    parcels = []
    labels = []
    labels_col = 'character'
    subj_idx = []
    
    for i, sub in enumerate(['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']):
        for segment in segments:
            parcels.append(path.join(parcels_root_path, f'sub-{sub}', 'shen_fconn', f'sub-{sub}_task-avmovie_run-{segment}_bold.csv'))
            labels.append(path.join(labels_root_path, f'emotions_av_1s_events_run-{segment}_events.tsv'))
            subj_idx.append(i)

    multi_label = []
    for segment in range(1,9):
        multi_label.append(pd.read_csv(path.join(labels_root_path, f'emotions_av_1s_events_run-{segment}_events.tsv'), sep='\t'))
    
    # Concat all labels and drop all exposures that are too short
    multi_label = pd.concat(multi_label)
    rows_to_include = multi_label.duration >= float(MIN_SEQ_LENGTH * SCANS_TR)
    # print(multi_label[rows_to_remove, labels_col].unique())
    # multi_label = multi_label.drop(multi_label.index[rows_to_remove])
    
    classes = multi_label[rows_to_include][labels_col].unique()
    label_map = {cls: i for (i, cls) in enumerate(classes)}
    
    return parcels, labels, labels_col, subj_idx, label_map


def train(training_set, validation_set, fold, args, subj_segment_sizes):
    model = TransformerDecoderClassifier(SHEN_PARCEL_DIM, args.embedding_dim, len(training_set.dataset.total_labels),
                                         args.n_attn_heads, args.rnn_num_layers)
    model = IndividualModelDecoder(subj_segment_sizes, SHEN_PARCEL_DIM, model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=args.lr_reduce_step_size)
    loss_fn = nn.CrossEntropyLoss()
    coach = Coach(model, optimizer, scheduler, loss_fn, training_set, validation_set, args.val_freq, args.max_epoch_len, fold)
    loss, acc = coach.train(args.num_epochs)
    return loss, acc, coach.model


def get_phase_model(training_folds, validation_folds, entire_train, outside_val, subj_segment_sizes, mode_label: str):
    args = get_args()

    fold_loss = []
    fold_acc = []

    for i, (train_set, val_set) in enumerate(zip(training_folds, validation_folds)):
        print(f"Training fold number {i+1}")
        loss, acc, _ = train(train_set, val_set, i + 1, args, subj_segment_sizes)
        fold_loss.append(loss)
        fold_acc.append(acc)
    print(f'{args.kfolds} folds: Mean loss={np.mean(fold_loss)}, mean acc={np.mean(fold_acc)}')

    mlflow.log_metric(f'{mode_label}, mean {len(training_folds)} loss', np.mean(fold_loss))
    mlflow.log_metric(f'{mode_label}, mean {len(training_folds)} acc', np.mean(fold_acc))

    loss, acc, model = train(entire_train, outside_val, 0, args, subj_segment_sizes)
    print(f'Entire training set, outside validation set: loss={loss}, acc={acc}')
    mlflow.log_metric(f'{mode_label}, outside validation loss', loss)
    mlflow.log_metric(f'{mode_label}, outside validation acc', acc)


def merge_labels_maps(phase1_map: Dict[str, int], phase2_map: Dict[str, int]) -> Dict[str, int]:
    labels_map = {}
    i = 0 
    for key in phase1_map:
        if key in phase2_map:
            labels_map[key] = i
            i += 1

    for key in phase2_map:
        if (key in phase1_map) and not (key in labels_map):
            labels_map[key] = i
            i += 1
    
    return labels_map


if __name__ == '__main__':
    print(EXPERIMENT_NAME)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=path.join(MLFLOW_ARTIFACT_STORE, EXPERIMENT_NAME))
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    run_name = None

    with mlflow.start_run(run_name=run_name):
        phase1_parcels, phase1_labels, labels_col, phase1_parcels_ids, phase1_labels_map = get_parcels_labels([1, 2, 3, 4])
        phase2_parcels, phase2_labels, labels_col, phase2_parcels_ids, phase2_labels_map = get_parcels_labels([5, 6, 7, 8])
        labels_map = merge_labels_maps(phase1_labels_map, phase2_labels_map)
        mlflow.log_param('classes', str(labels_map))
        
        args = get_args()
        phase1_train_sets, phase1_val_sets, phase1_entire_set, subj_segment_sizes = get_kfolds(args.kfolds, args.batch_size, phase1_parcels, phase1_parcels_ids,
                                                    phase1_labels, labels_map, labels_col)

        
        phase2_train_sets, phase2_val_sets, phase2_entire_set, subj_segment_sizes = get_kfolds(args.kfolds, args.batch_size, phase2_parcels, phase2_parcels_ids,
                                                                    phase2_labels, labels_map, labels_col)
        print("Train on first half")
        get_phase_model(phase1_train_sets, phase1_val_sets, phase1_entire_set, phase2_entire_set, subj_segment_sizes, 'Train on first half')
        print("Train on second half")
        get_phase_model(phase2_train_sets, phase2_val_sets, phase2_entire_set, phase1_entire_set, subj_segment_sizes, 'Train on second half')
