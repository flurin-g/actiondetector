import json
from datetime import date
from itertools import chain
from pathlib import Path
from typing import List, Tuple

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from nltk import everygrams
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

from src.base_logger import logger, ROOT_PATH
from src.utils import read_json, write_to_disk
from src.word_stats import ActionItemState

K_SPLITS = 5


def eval_is_action_item(previous_is_action: str, next_is_action: str) -> str:
    if previous_is_action == "yes" or next_is_action == "yes":
        return "yes"
    else:
        return "no"


def extend_utterance(last_utterance: dict, utterance: dict):
    last_utterance["text"] += " " + (utterance["text"])
    last_utterance["isActionItem"] = eval_is_action_item(last_utterance["isActionItem"], utterance["isActionItem"])

    last_utterance["features"][0].extend(utterance["features"][0])
    last_utterance["features"][1].extend(utterance["features"][1])

    return last_utterance


def merge_utterances_by_same_speaker(utterances_of_transcript):
    merged_utterances = [utterances_of_transcript.pop(0)]
    for utterance in utterances_of_transcript:
        if merged_utterances[-1]["speaker"] == utterance["speaker"]:
            merged_utterances[-1:] = [extend_utterance(merged_utterances[-1], utterance)]
        else:
            merged_utterances.append(utterance)
    return merged_utterances


def write_merged_utterances(src_folder: Path, trgt_path: Path):
    file_list = list(src_folder.glob('*.json'))

    transcripts = list()
    for file in file_list:
        utterances_of_transcript = read_json(file)
        transcripts.append(merge_utterances_by_same_speaker(utterances_of_transcript))
    with open(trgt_path, 'w') as f:
        f.write(json.dumps(transcripts))


def load_utterances_flat(src_path: Path) -> List[dict]:
    merged_utterances = read_json(src_path)
    return list(chain.from_iterable(merged_utterances))


def is_action_item(utterance):
    return utterance["isActionItem"] == ActionItemState.YES.value


def calculate_utterance_ratios(utterances: List[dict], trgt_path):
    num_total_utterances = len(utterances)
    num_pos_utterances = len([utterance for utterance in utterances if is_action_item(utterance)])
    num_neg_utterances = num_total_utterances - num_pos_utterances
    ratios = f'# Utterance Ratios\n' \
             f'- Total number of utterances: {num_total_utterances}\n' \
             f'- Number of positive utterances: {num_pos_utterances}' \
             f'\t{num_pos_utterances / float(num_total_utterances)} %\n' \
             f'- Number of negative utterances: {num_neg_utterances}' \
             f'\t{num_neg_utterances / float(num_total_utterances)} %'
    logger.info(ratios)
    with open(trgt_path, "w") as f:
        f.write(ratios)


def fetch_tokens(token_type: str, utterance):
    return [token[token_type] for token in utterance["features"][0]]


def fetch_entity(utterance):
    entities = [ent["entity"] for ent in utterance["features"][1]]
    return entities if entities else ["NONE"]


def to_binary_labels(param):
    return 1 if param == "yes" else 0


def extract_features(utterances: List[dict]):
    return zip(*[((fetch_tokens("lemma", utterance),
                   fetch_tokens("pos", utterance),
                   fetch_entity(utterance)),
                  to_binary_labels(utterance["isActionItem"])) for utterance in utterances])


class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, mode: str, ngram_range: tuple = None):
        """
        :param ngram_range:
        :param mode: mode can either be "text" return tokens verbatim text
                                        "string" return the concatenated string
                                        "pos": return part of speech tags
        """
        self.mode = mode
        self.ngram_range = ngram_range

    def make_ngrams(self, X: list):
        X_res = [everygrams(doc, self.ngram_range[0], self.ngram_range[1])
                 for doc in X]
        return X_res

    def fit(self, X, y=None):
        return self

    def transform(self, X: list):
        if self.mode == "lemma":
            val, _, _ = zip(*X)
            val = self.make_ngrams(val)
        elif self.mode == "pos":
            _, val, _ = zip(*X)
        elif self.mode == "ent":
            _, _, val = zip(*X)
            val = self.make_ngrams(val)
        return val


def create_classifier(mode: str, ngram_range: Tuple[int, int]):
    return Pipeline([('feature_selector', FeatureSelector(mode=mode, ngram_range=ngram_range)),
                     ('tf_idf', TfidfVectorizer(preprocessor=lambda x: x,
                                                tokenizer=lambda x: x,
                                                min_df=2,
                                                use_idf=True,
                                                sublinear_tf=True)),
                     ('svm', SVC())
                     ])


def create_pos_classifier():
    return Pipeline([
        ('pos_selector', FeatureSelector(mode="pos", ngram_range=(2, 3))),
        ('tf_idf', TfidfVectorizer(preprocessor=lambda x: x,
                                   tokenizer=lambda x: x,
                                   min_df=2, use_idf=True, sublinear_tf=True)),
        ('svm', SVC())

    ])


def create_stacked_ensemble():
    estimators = [
        ('lemma', make_pipeline(FeatureSelector(mode="lemma", ngram_range=(1, 3)),
                                TfidfVectorizer(preprocessor=lambda x: x,
                                                tokenizer=lambda x: x,
                                                min_df=2, use_idf=True, sublinear_tf=True),
                                SVC())),
        ('pos', make_pipeline(FeatureSelector(mode="pos", ngram_range=(2, 3)),
                              TfidfVectorizer(preprocessor=lambda x: x,
                                              tokenizer=lambda x: x,
                                              min_df=2, use_idf=True, sublinear_tf=True),
                              SVC()))
    ]
    return StackingClassifier(estimators, final_estimator=SVC())


def create_stacked_ensemble_with_ent():
    estimators = [
        ('lemma', make_pipeline(FeatureSelector(mode="lemma", ngram_range=(1, 3)),
                                TfidfVectorizer(preprocessor=lambda x: x,
                                                tokenizer=lambda x: x,
                                                min_df=2, use_idf=True, sublinear_tf=True),
                                SVC())),
        ('pos', make_pipeline(FeatureSelector(mode="pos", ngram_range=(2, 3)),
                              TfidfVectorizer(preprocessor=lambda x: x,
                                              tokenizer=lambda x: x,
                                              min_df=2, use_idf=True, sublinear_tf=True),
                              SVC())),
        ('ent', make_pipeline(FeatureSelector(mode="ent", ngram_range=(2, 3)),
                              CountVectorizer(preprocessor=lambda x: x,
                                              tokenizer=lambda x: x),
                              SVC()))
    ]
    return StackingClassifier(estimators, final_estimator=SVC())


def k_fold_evaluation(clf, X_train, y_train, scoring_method: str):
    skf = StratifiedKFold(n_splits=K_SPLITS, shuffle=True, random_state=42)
    return cross_val_score(clf, X_train, y_train, scoring=scoring_method, cv=skf)


def prepare_datasets(src_path):
    utterances = load_utterances_flat(src_path)
    X, y = extract_features(utterances)
    X = np.array(X, dtype=object)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    undersampler = RandomUnderSampler(sampling_strategy='majority')

    X_over, y_over = undersampler.fit_resample(X_train, y_train)
    return X_train, X_over, X_test, y_train, y_over, y_test


def evaluate_clf(clf, X, y, X_test, y_test, experiment_name: str):
    scoring = ["accuracy", "f1"]
    k_fold_log = "\n"
    for score in scoring:
        metric = k_fold_evaluation(clf, X, y, score)
        k_fold_log += f'Cross validated {score} score: {np.mean(metric)}\n'
    logger.info(k_fold_log)

    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    class_report_res = "Classification report:\n```\n" + classification_report(y_test, y_pred,
                                                                               target_names=("no", "yes")) + "```"
    logger.info(class_report_res)

    return "# " + experiment_name + k_fold_log + class_report_res


def run_classifier(training_data_path: Path, mode: str):
    X_train, X_under, X_test, y_train, y_under, y_test = prepare_datasets(training_data_path)

    if mode == "lemma":
        clf = create_classifier(mode="lemma", ngram_range=(1, 3))
    elif mode == "stacked":
        clf = create_stacked_ensemble()
    elif mode == "stacked-ent":
        clf = create_stacked_ensemble_with_ent()

    evaluation_string_regular = evaluate_clf(clf, X_train, y_train, X_test, y_test, "Regular")

    evaluation_string_under = evaluate_clf(clf, X_under, y_under, X_test, y_test, "Undersampled")

    write_to_disk(evaluation_string_regular + "\n" + evaluation_string_under,
                  ROOT_PATH / Path("data") / f'{date.today()}-evaluation',
                  "md")
