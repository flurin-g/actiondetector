from datetime import date
from enum import Enum
from functools import partial
from itertools import chain
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from collections import Counter
import en_core_web_lg

from src.base_logger import ROOT_PATH
from src.utils import read_json

nlp = en_core_web_lg.load()

PLOT_FOLDER = "plots"
MAX_FEATURES = 12
PLOT_FONT_SIZE = 24
PLOT_TITLE_FONT_SIZE = 26
FIGSIZE = (10, 10)

# plt.rcParams.update({'font.size': PLOT_FONT_SIZE})
plt.rc('ytick', labelsize=PLOT_FONT_SIZE)
plt.rc('axes', titlesize=PLOT_TITLE_FONT_SIZE)


def init_nltk():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')


class ActionItemState(Enum):
    YES = "yes"
    NO = "no"
    MAYBE = "maybe"
    ALL = "all"

    def __str__(self):
        return self.value


class Feature(Enum):
    TEXT = "text"
    LEMMA = "lemma"
    POS = "pos"
    ENT = "ent"

    def __str__(self):
        return self.value


def filter_action_items(filter_by: ActionItemState, is_action_item: str):
    return filter_by.value == is_action_item or filter_by == ActionItemState.ALL


def feature_extract(text: str, to_lower: bool = True) -> Tuple[List[dict], list]:
    doc = nlp(text)
    return [{
        "text": token.text.lower() if to_lower else token.text,
        "lemma": token.lemma_,
        "pos": token.pos_,
        "isStop": token.is_stop
    } for token in doc], [{
        "text": ent.text,
        "entity": ent.label_
    } for ent in doc.ents]


def extract_features_by_action_state(utterances, filter_items: ActionItemState):
    return [utterance["features"]
            for utterance in utterances
            if filter_action_items(filter_items, utterance["isActionItem"])]


def fetch_feature_container(container: list, feature: Feature) -> Union[dict, list]:
    if feature == Feature.ENT:
        return container[1]
    else:
        return container[0]


def extract_token_and_flatten(utterances: List[List[dict]], feature: Feature, action_state: ActionItemState):
    return chain.from_iterable([
        fetch_feature_container(container, feature)
        for container in extract_features_by_action_state(utterances, filter_items=action_state)
    ])


def plot_all_token_counts(src_folder: Path):
    file_list = list(src_folder.glob('*.json'))
    utterances = list()
    for file in file_list:
        utterances = [*utterances, *read_json(file)]

    for feature in Feature:
        plot_by_feature(utterances, feature)


def format_title(feature: Feature, action_state: ActionItemState) -> str:
    return f'{feature.value.capitalize()} counts, Action item: {action_state.value}'


def plot_by_feature(utterances, feature: Feature):
    extractor = partial(extract_token_and_flatten, utterances=utterances, feature=feature)
    plotting_function = plotting_functions[feature]

    for acn_state in [ActionItemState.ALL, ActionItemState.YES, ActionItemState.NO]:
        plotting_function(extractor(action_state=acn_state), title=format_title(feature, acn_state))


remove_list = ["yeah", "um", "uh", "uh-", "mmhmm", "hmm", "mm", "oh", "ok", "o_k", "o_k.", "na", "nt", "gon",
               "so", "s", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "-",
               "_", "o", "o_-", "th-", "s-", "w-", "n-", "wh-", "p-", "i-", "y-", "ye-", "ind-", " "]


def remove_specified_words(utterance: List[dict]):
    return [token for token in utterance if not token["text"] in remove_list]


def remove_stop_words(utterance: List[dict]):
    return [token for token in utterance if not token["isStop"]]


def remove_punct_marks(utterance: List[dict]):
    return [token for token in utterance if not token["pos"] == "PUNCT"]


def map_to_token(utterance: List[dict], key: str, remove_stop: bool = True, remove_punct: bool = True) -> List[str]:
    utterance = remove_specified_words(utterance)
    if remove_stop:
        utterance = remove_stop_words(utterance)
    if remove_punct:
        utterance = remove_punct_marks(utterance)
    return [token[key] for token in utterance]


map_to_text = partial(map_to_token, key="text")

map_to_pos = partial(map_to_token, key="pos")

map_to_lemma = partial(map_to_token, key="lemma")

map_to_entity = partial(map_to_token, key="entity", remove_stop=False, remove_punct=False)


def extract_sorted_xy_coordinates(word_counts: dict) -> Tuple[np.ndarray, np.ndarray]:
    X = list()
    y = list()
    for key in word_counts.keys():
        X.append(key)
        y.append(word_counts[key])
    X = np.array(X)
    y = np.array(y)
    indices = np.argsort(y)[::-1]
    return X[indices], y[indices]


def plot_token_counts(token_counts: dict, title: str) -> None:
    X, y = extract_sorted_xy_coordinates(token_counts)
    sns.barplot(y, X).set_title(title)
    plt.savefig(ROOT_PATH / Path(PLOT_FOLDER) / f'{date.today()}, {title}.png')
    plt.show()


def plot_counts(texts: List[list[dict]], mapping_function: callable, title: str) -> None:
    tokens = [text for text in mapping_function(texts)]
    counts = Counter(tokens)
    count_df = pd.DataFrame(counts.most_common(MAX_FEATURES),
                            columns=['words', 'count'])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    count_df.sort_values(by='count').plot.barh(x='words',
                                               y='count',
                                               ax=ax,
                                               color="purple")

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(ROOT_PATH / Path(PLOT_FOLDER) / f'{date.today()}, {title}.png')
    plt.show()


plot_word_counts = partial(plot_counts, mapping_function=map_to_text)

plot_pos_counts = partial(plot_counts, mapping_function=map_to_pos)

plot_lemma_counts = partial(plot_counts, mapping_function=map_to_lemma)

plot_entity_counts = partial(plot_counts, mapping_function=map_to_entity)

plotting_functions = {
    Feature.TEXT: plot_word_counts,
    Feature.POS: plot_pos_counts,
    Feature.LEMMA: plot_lemma_counts,
    Feature.ENT: plot_entity_counts
}


def parse_icsi_json(transcript: dict) -> dict:
    transcript["features"] = feature_extract(transcript["text"])
    return transcript
