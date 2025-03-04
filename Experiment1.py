from argparse import ArgumentParser
from models import TransformerDecoderClassifier, IndividualizedModel, MLP
from training import Coach
from typing import List, Tuple, Dict
from os import path
from consts import SHEN_PARCEL_DIM, MLFLOW_ARTIFACT_STORE, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MIN_SEQ_LENGTH, SCANS_TR, LABELS_ROOT_PATH, PARCELS_ROOT_PATH
from data import get_kfolds, merge_consecutive_labels
from torch import optim, nn
import numpy as np
import pandas as pd
import mlflow


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=512, help='The hidden representation dimension')
    parser.add_argument('--kfolds', type=int, default=5, help='number of folds to divide the training data (KFold validation)')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of samples per batch')
    parser.add_argument('--n_attn_heads', type=int, default=4, help='Number of attention heads to use (if transformer)')
    parser.add_argument('--lr', type=int, default=1e-5, help='The initial LR (using Adam optimizer)')
    parser.add_argument('--lr_reduce_step_size', type=int, default=18, help='After how many epochs should the LR be reduced?')
    parser.add_argument('--val_freq', type=int, default=10, help='Validation set epoch frequency')
    parser.add_argument('--max_epoch_len', type=int, default=1000, help='Maximum number of batches in an epoch')
    parser.add_argument('--num_epochs', type=int, default=20, help='Num epochs to train the model on')
    parser.add_argument('--num_layers', type=int, default=4, help='Num RABs / ReLU(Linear) to use in the decoder / MLP')
    parser.add_argument('--use_mlp', action='store_true', help='Train an MLP instead of a Transformer Decoder')
    parser.add_argument('--run_name', type=str, default=None, help='Train an MLP instead of a Transformer Decoder')

    return parser.parse_args()


def get_parcels_labels(segments: List[int]) -> Tuple[List[str], List[str], str, List[int], Dict[str, int]]:
    """
    Given a list of segment indices, return the segments for the first experiment
    :param segments: A list of segment indices
    :return: (List of Shen parcelations file paths, List of labels file paths, Label column)
    """
    parcels_root_path = PARCELS_ROOT_PATH
    labels_root_path = LABELS_ROOT_PATH
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
    for segment in segments:
        curr_segment = pd.read_csv(path.join(labels_root_path, f'emotions_av_1s_events_run-{segment}_events.tsv'), sep='\t')
        curr_segment = merge_consecutive_labels(curr_segment, labels_col)
        multi_label.append(curr_segment)
    
    # Concat all labels and drop all exposures that are too short
    multi_label = pd.concat(multi_label)
    rows_to_include = multi_label.duration >= float(MIN_SEQ_LENGTH * SCANS_TR)
    # print(multi_label[rows_to_remove, labels_col].unique())
    # multi_label = multi_label.drop(multi_label.index[rows_to_remove])
    
    classes = multi_label[rows_to_include][labels_col].unique()
    label_map = {cls: i for (i, cls) in enumerate(classes)}
    
    return parcels, labels, labels_col, subj_idx, label_map


def train(training_set, validation_set, fold, args, subj_segment_sizes, label):
    mlflow.log_params(vars(args))
    if not args.use_mlp:
        model = TransformerDecoderClassifier(SHEN_PARCEL_DIM, args.embedding_dim, len(training_set.dataset.total_labels),
                                            args.n_attn_heads, args.num_layers)
    else:
        model = MLP(SHEN_PARCEL_DIM, hidden_dim=args.embedding_dim, num_cls=len(training_set.dataset.total_labels), num_layers=args.num_layers)
    model = IndividualizedModel(subj_segment_sizes, SHEN_PARCEL_DIM, model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=args.lr_reduce_step_size)
    loss_fn = nn.CrossEntropyLoss()
    coach = Coach(model, optimizer, scheduler, loss_fn, training_set, validation_set, args.val_freq, args.max_epoch_len, label, fold)
    loss, acc = coach.train(args.num_epochs)
    return loss, acc, coach.model


def get_phase_model(training_folds, validation_folds, entire_train, outside_val, subj_segment_sizes, mode_label: str):
    args = get_args()

    fold_loss = []
    fold_acc = []

    for i, (train_set, val_set) in enumerate(zip(training_folds, validation_folds)):
        print(f"Training fold number {i+1}")
        loss, acc, _ = train(train_set, val_set, i + 1, args, subj_segment_sizes, mode_label)
        fold_loss.append(loss)
        fold_acc.append(acc)
    print(f'{args.kfolds} folds: Mean loss={np.mean(fold_loss)}, mean acc={np.mean(fold_acc)}')

    mlflow.log_metric(f'{mode_label} mean {len(training_folds)} fold loss', np.mean(fold_loss))
    mlflow.log_metric(f'{mode_label} mean {len(training_folds)} fold acc', np.mean(fold_acc))

    loss, acc, model = train(entire_train, outside_val, 0, args, subj_segment_sizes, mode_label)
    print(f'Entire training set, outside validation set: loss={loss}, acc={acc}')
    mlflow.log_metric(f'{mode_label} outside validation loss', loss)
    mlflow.log_metric(f'{mode_label} outside validation acc', acc)


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
    
    # run_name = 'Stronger overfit on curr half'
    args = get_args()

    with mlflow.start_run(run_name=args.run_name):
        phase1_parcels, phase1_labels, labels_col, phase1_parcels_ids, phase1_labels_map = get_parcels_labels([1, 2, 3, 4])
        phase2_parcels, phase2_labels, labels_col, phase2_parcels_ids, phase2_labels_map = get_parcels_labels([5, 6, 7, 8])
        labels_map = merge_labels_maps(phase1_labels_map, phase2_labels_map)
        mlflow.log_param('classes', str(labels_map))
        
        
        max_seq_length = -1
        if args.use_mlp:
            max_seq_length = 1

        phase1_train_sets, phase1_val_sets, phase1_entire_set, subj_segment_sizes = get_kfolds(args.kfolds, args.batch_size, phase1_parcels, phase1_parcels_ids,
                                                    phase1_labels, labels_map, labels_col, max_seq_length=max_seq_length, use_mlp=args.use_mlp)

        
        phase2_train_sets, phase2_val_sets, phase2_entire_set, subj_segment_sizes = get_kfolds(args.kfolds, args.batch_size, phase2_parcels, phase2_parcels_ids,
                                                                    phase2_labels, labels_map, labels_col, max_seq_length=max_seq_length, use_mlp=args.use_mlp)
        print("Train on first half")
        get_phase_model(phase1_train_sets, phase1_val_sets, phase1_entire_set, phase2_entire_set, subj_segment_sizes, 'Train on first half')
        print("Train on second half")
        get_phase_model(phase2_train_sets, phase2_val_sets, phase2_entire_set, phase1_entire_set, subj_segment_sizes, 'Train on second half')
