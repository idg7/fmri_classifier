import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict
from .consts import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def get_human_model(rdms: List[pd.DataFrame]) -> pd.DataFrame:
    labels = rdms[0].columns
    rdms = [np.expand_dims(df.to_numpy(), 0) for df in rdms]
    rdms = np.concatenate(rdms, axis=0)

    model = np.mean(rdms, axis=0)

    return pd.DataFrame(model, columns=labels, index=labels)


def model_correlations(human: pd.DataFrame, model: pd.DataFrame) -> float:
    labels = human.columns
    y = human.to_numpy().flatten()
    filtered_model = model.loc[labels, labels]
    x = filtered_model.to_numpy().flatten()
    return stats.pearsonr(x, y).statistic


def plot(df: pd.DataFrame, num_char_in_segment: List[int], information_cols: List[str]) -> None:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for info in information_cols:
        fig.add_trace(
            go.Scatter(
                y=df[info], x=df['segment_idx'], name=info))


    fig.add_trace(
        go.Bar(
            y=num_char_in_segment, 
            x=[seg for seg in SEGMENT_IDS], name='Num characters', opacity=0.2),
            secondary_y=True)

    fig.update_yaxes(title_text="Correlations", secondary_y=False)
    fig.update_yaxes(title_text="Num characters in segment", secondary_y=True)
    fig.show()


def models_comparison(segment_subj_rdms: List[List[pd.DataFrame]], models: Dict[str, pd.Dataframe]) -> None:
    num_char_in_segment = [segment_subj_rdms[i][0].shape[0] for i, _ in enumerate(SEGMENT_IDS)]

    all_comparisons = {'segment_idx': [seg for i, seg in enumerate(SEGMENT_IDS) if num_char_in_segment[i] > 2]}

    for info in models:
        all_comparisons[info] = []
    
    for i, segment in enumerate(SEGMENT_IDS):
        if num_char_in_segment[i] > 2:
            human_model = get_human_model(segment_subj_rdms[i])

            for info in models:
                corr = model_correlations(human_model, models[info])
                all_comparisons[info].append(corr)
    
    plot(pd.DataFrame(all_comparisons), num_char_in_segment, [info for info in models])
            