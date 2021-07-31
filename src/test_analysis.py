from collections import Counter
from pathlib import Path
from unittest import TestCase

import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from src.analysis import write_merged_utterances, merge_utterances_by_same_speaker, is_action_item, run_classifier, \
    extract_features, FeatureSelector, load_utterances_flat, prepare_datasets, create_classifier, k_fold_evaluation, \
    evaluate_clf, create_stacked_ensemble
from src.base_logger import ROOT_PATH, logger
from src.utils import read_json, MERGED_UTTERANCES_PATH

DEMO_UTTERANCE_POS = {
    "speaker": "me012",
    "text": "Nice",
    "isActionItem": "yes",
    "features": [
        [
            {
                "text": "nice",
                "lemma": "nice",
                "pos": "ADJ",
                "isStop": False
            }
        ],
        []
    ]
}

DEMO_UTTERANCE_NEG = {
    "speaker": "me012",
    "text": "Not nice",
    "isActionItem": "no",
    "features": [
        [
            {
                "text": "nice",
                "lemma": "nice",
                "pos": "ADJ",
                "isStop": False
            }
        ],
        []
    ]
}

DEMO_UTTERANCES = [
    {
        "speaker": "me012",
        "text": "Nice",
        "isActionItem": "yes",
        "features": [
            [
                {
                    "text": "nice",
                    "lemma": "nice",
                    "pos": "ADJ",
                    "isStop": False
                }
            ],
            []
        ]
    },
    {
        "speaker": "fe004",
        "text": "OK",
        "isActionItem": "no",
        "features": [
            [
                {
                    "text": "ok",
                    "lemma": "ok",
                    "pos": "INTJ",
                    "isStop": False
                }
            ],
            []
        ]
    },
    {
        "speaker": "mn015",
        "text": "to to handle",
        "isActionItem": "no",
        "features": [
            [
                {
                    "text": "to",
                    "lemma": "to",
                    "pos": "PART",
                    "isStop": True
                },
                {
                    "text": "to",
                    "lemma": "to",
                    "pos": "PART",
                    "isStop": True
                },
                {
                    "text": "handle",
                    "lemma": "handle",
                    "pos": "VERB",
                    "isStop": False
                }
            ],
            [
                {
                    "text": "many months",
                    "entity": "DATE"
                },
                {
                    "text": "london bridge",
                    "entity": "PLACE"
                }
            ]
        ]
    }
]


class TestAnalysis(TestCase):
    def test_load_all_transcripts(self):
        write_merged_utterances(ROOT_PATH / Path("test/test_data/extracted_features"), 'data/')

    def test_merge_utterances_by_same_speaker(self):
        utterances_of_transcript = read_json(ROOT_PATH / Path("test/test_data/extracted_features/Bed003.json"))
        res = merge_utterances_by_same_speaker(utterances_of_transcript)
        print(res)

    def test_is_action_item(self):
        res = is_action_item(DEMO_UTTERANCE_POS)
        print(res)

    def test_extract_features(self):
        X, y = extract_features(DEMO_UTTERANCES)
        print(X)
        print("-------")
        print(y)

    def test_feature_selector(self):
        utterances = load_utterances_flat(ROOT_PATH / Path(MERGED_UTTERANCES_PATH))
        X, y = extract_features(utterances)

        fs = FeatureSelector(mode="lemma", ngram_range=(1, 3))
        res = fs.transform(X)
        for elem in res:
            print(res)

    def test_evaluate_clf(self):
        X_train, X_under, X_test, y_train, y_under, y_test = prepare_datasets(ROOT_PATH / Path(MERGED_UTTERANCES_PATH))
        clf = create_classifier(mode="lemma", ngram_range=(1, 3))

        X_under = X_under[:5_000]
        y_under = y_under[:5_000]

        evaluation_string = evaluate_clf(clf, X_under, y_under, X_test, y_test, "Undersampled")

        print(f"Start of evaluation String:\n{evaluation_string}")

    def test_stacked_clf(self):
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
        clf = StackingClassifier(estimators, final_estimator=LogisticRegression())

        X_train, X_under, X_test, y_train, y_under, y_test = prepare_datasets(ROOT_PATH / Path(MERGED_UTTERANCES_PATH))

        X_under = X_under[:5_000]
        y_under = y_under[:5_000]

        evaluation_string = evaluate_clf(clf, X_under, y_under, X_test, y_test, "Undersampled")
