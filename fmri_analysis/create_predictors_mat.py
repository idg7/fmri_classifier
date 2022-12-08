from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from .consts import *
from .create_predictors_mat import *


def create_label_vector(labels: pd.DataFrame, character: str, num_TRs: int) -> np.ndarray:
    # Filter just the target label
    character_shows = labels.loc[labels[LABEL_COL] == character]
    # Add start and end TR indices
    character_shows = character_shows.assign(start_tr = character_shows['onset'] // TR)
    character_shows = character_shows.assign(end_tr = character_shows['start_tr'] + 1 + (character_shows['duration'] // TR))
    
    # build a binary character appearance vector, filled with 0s when the character isn't shown, and 1s when it is shown
    character_appearance = np.zeros(num_TRs)
    for idx, row in character_shows.iterrows():
        character_appearance[int(row['start_tr']) : int(row['end_tr'])] = 1
    y = np.expand_dims(character_appearance, 0)

    return y


def create_character_level_label_vector(labels: pd.DataFrame, character: str, num_TRs: int) -> np.ndarray:
    # Add start and end TR indices
    labels = labels.assign(start_tr = labels['onset'] // TR)
    labels = labels.assign(end_tr = labels['start_tr'] + 1 + (labels['duration'] // TR))
    
    # Filter the target label (positive)
    positive = labels.loc[labels[LABEL_COL] == character]
    negative = labels.loc[labels[LABEL_COL] != character]
    
    
    # build a binary character appearance vector, filled with 0s when the character isn't shown, and 1s when it is shown
    character_appearance = np.zeros(num_TRs) - 1
    for idx, row in positive.iterrows():
        character_appearance[int(row['start_tr']) : int(row['end_tr'])] = 1

    for idx, row in negative.iterrows():
        character_appearance[int(row['start_tr']) : int(row['end_tr'])] = 0
    
    y = np.expand_dims(character_appearance, 0)

    return y


def create_character_one_hot_vector(labels: pd.DataFrame, characters: List[str], num_TRs: int) -> pd.DataFrame:
    x = {}
    for character in characters:
        x_char = create_label_vector(labels, character, num_TRs).squeeze(0)
        X[character] = x_char
    x = pd.DataFrame(X)

    return x