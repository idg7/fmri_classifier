from sklearn.model_selection import LeaveOneOut
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from typing import List
from .consts import *


def lower_noise_ceiling(rdms: List[pd.DataFrame]) -> List[float]:
    tril = np.tril_indices(rdms[0].shape[0], -1)
    corrs = []
    
    rdms = [np.expand_dims(df.to_numpy(), 0) for df in rdms]
    rdms = np.concatenate(rdms, axis=0)
    
    loo = LeaveOneOut()

    for train_index, test_index in loo.split(rdms):
        loo_rdm = rdms[test_index][0]
        model = rdms[train_index]
        model = np.mean(model, axis=0)
        corrs.append(pearsonr(loo_rdm[tril], model[tril]).statistic)

    return corrs


def upper_noise_ceiling(rdms: List[pd.DataFrame]) -> List[float]:
    tril = np.tril_indices(rdms[0].shape[0], -1)
    corrs = []
    
    rdms = [np.expand_dims(df.to_numpy(), 0) for df in rdms]
    rdms = np.concatenate(rdms, axis=0)

    model = np.mean(rdms, axis=0)
    for i in range(rdms.shape[0]):
        rdm = rdms[i]
        corrs.append(pearsonr(rdm[tril], model[tril]).statistic)

    return corrs


def plot(df: pd.DataFrame, num_char_in_segment: List[int]) -> None:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Violin(y=df['value'], box_visible=True,
                                meanline_visible=True, opacity=0.6,
                                x=df['variable']))

    for subj in VALID_SUBJ_IDS:
        subj_df = df[df['Subject'] == subj]
        fig.add_trace(go.Scatter(y=subj_df['value'], x=subj_df['variable'], name=f'Subject {subj}'))

    means = df.groupby('variable').mean()
    fig.add_trace(
        go.Scatter(
            y=means['value'], x=means.index, name='MEANS'))


    fig.add_trace(
        go.Bar(
            y=num_char_in_segment, 
            x=[seg for seg in SEGMENT_IDS], name='Num characters', opacity=0.2),
            secondary_y=True)

    fig.update_yaxes(title_text="Correlations", secondary_y=False)
    fig.update_yaxes(title_text="Num characters in segment", secondary_y=True)
    fig.show()


def noise_ceiling_segments(segment_subj_rdms: List[List[pd.DataFrame]], upper: bool = False) -> None:
    segment_corrs = {'Subject': VALID_SUBJ_IDS}
    for i, segment in enumerate(SEGMENT_IDS):
        n_i = segment_subj_rdms[i][0].shape[0]
        if n_i > 2:
            if upper:
                segment_corrs[segment] = upper_noise_ceiling(segment_subj_rdms[i])
            else:
                segment_corrs[segment] = lower_noise_ceiling(segment_subj_rdms[i])

    df = pd.DataFrame(segment_corrs).melt(id_vars='Subject')

    plot(df, [segment_subj_rdms[i][0].shape[0] for i, _ in enumerate(SEGMENT_IDS)])

