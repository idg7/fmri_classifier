from argparse import ArgumentParser
from models import TransformerDecoderClassifier
from training import Coach
from typing import List, Tuple
from os import path
from consts import SHEN_PARCEL_DIM
from data import get_kfolds
from torch import optim, nn
import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument('embedding_dim', type=int, default=512)
    parser.add_argument('kfolds', type=int, default=5)
    parser.add_argument('batch_size', type=int, default=16)
    parser.add_argument('n_attn_heads', type=int, default=8)
    parser.add_argument('lr', type=int, default=8)
    parser.add_argument('lr_reduce_step_size', type=int, default=80)
    parser.add_argument('val_freq', type=int, default=10)
    parser.add_argument('max_epoch_len', type=int, default=100)
    parser.add_argument('num_epochs', type=int, default=100)

    return parser.parse_args()


def get_parcels_labels(segments: List[int]) -> Tuple[List[str], List[str], str]:
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
    for sub in ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']:
        for segment in segments:
            parcels.append(path.join(parcels_root_path, f'sub-{sub}', 'shen_fconn', f'sub-{sub}_task-avmovie_run-{segment}_bold.csv'))
            labels.append(path.join(labels_root_path, f'conf_audio_run-{segment}_events.tsv'))
    return parcels, labels, labels_col


def train(training_set, validation_set, fold, args):
    model = TransformerDecoderClassifier(SHEN_PARCEL_DIM, args.embedding_dim, len(training_set.dataset.total_labels),
                                         args.n_attn_heads)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=args.lr_reduce_step_size)
    loss_fn = nn.CrossEntropyLoss()
    coach = Coach(model, optimizer, scheduler, loss_fn, training_set, validation_set, args.val_freq, args.max_epoch_len, fold)
    loss, acc = coach.train(args.num_epochs)
    return loss, acc, coach.model


def get_phase_model(training_folds, validation_folds, entire_train, outside_val):
    args = get_args()

    fold_loss = []
    fold_acc = []

    for i, (train_set, val_set) in enumerate(zip(training_folds, validation_folds)):
        loss, acc, _ = train(train_set, val_set, i + 1, args)
        fold_loss.append(loss)
        fold_acc.append(acc)
    print(f'{args.kfolds} folds: Mean loss={np.mean(fold_loss)}, mean acc={np.mean(fold_loss)}')

    loss, acc, model = train(entire_train, outside_val, 0, args)
    print(f'Entire training set, outside validation set: loss={loss}, acc={acc}')


if __name__ == '__main__':
    phase1_parcels, phase1_labels, labels_col = get_parcels_labels([1, 2, 3, 4])
    args = get_args()
    phase1_train_sets, phase1_val_sets, phase1_entire_set = get_kfolds(args.kfolds, args.batch_size, phase1_parcels,
                                                  phase1_labels, labels_col)

    phase2_parcels, phase2_labels, labels_col = get_parcels_labels([5,6,7,8])
    phase2_train_sets, phase2_val_sets, phase2_entire_set = get_kfolds(args.kfolds, args.batch_size, phase2_parcels,
                                                                phase2_labels, labels_col)
    print("Train on first half")
    get_phase_model(phase1_train_sets, phase1_val_sets, phase1_entire_set, phase2_entire_set)
    print("Train on second half")
    get_phase_model(phase2_train_sets, phase2_val_sets, phase2_entire_set, phase1_entire_set)
