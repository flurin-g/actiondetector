from pathlib import Path
from unittest import TestCase

from src.base_logger import ROOT_PATH
from src.word_stats import ActionItemState, extract_sorted_xy_coordinates, plot_token_counts, plot_pos_counts, feature_extract, map_to_pos, map_to_entity, map_to_text, \
    extract_features_by_action_state
from src.utils import read_json
from src.word_stats import filter_action_items

DEMO_TRANSCRIPT = [
    {
        "speaker": "me026",
        "text": "Channel one.",
        "isActionItem": "no"
    },
    {
        "speaker": "mn007",
        "text": "Test.",
        "isActionItem": "yes"
    },
    {
        "speaker": "fn002",
        "text": "Hello.",
        "isActionItem": "no"
    },
    {
        "speaker": "me026",
        "text": "Channel three.",
        "isActionItem": "yes"
    }
]

DEMO_FEATURES = [
    {'text': 'we', 'lemma': '-PRON-', 'pos': 'PRON', 'isStop': True},
    {'text': 'need', 'lemma': 'need', 'pos': 'VERB', 'isStop': False},
    {'text': 'to', 'lemma': 'to', 'pos': 'PART', 'isStop': True},
    {'text': 'know', 'lemma': 'know', 'pos': 'VERB', 'isStop': False},
    {'text': 'by', 'lemma': 'by', 'pos': 'ADP', 'isStop': True},
    {'text': 'the', 'lemma': 'the', 'pos': 'DET', 'isStop': True},
    {'text': 'fifth', 'lemma': 'fifth', 'pos': 'NOUN', 'isStop': False},
    {'text': 'of', 'lemma': 'of', 'pos': 'ADP', 'isStop': True},
    {'text': 'april', 'lemma': 'April', 'pos': 'PROPN', 'isStop': False},
    {'text': ',', 'lemma': ',', 'pos': 'PUNCT', 'isStop': False}
]

WORD_COUNT_DICT = {
    "the": 12,
    "lazy": 9,
    "hog": 122,
    "jumped": 52,
    "over": 7
}

DEMO_SENTENCE = "You dont need to read the digits if we think that thats torture but the reason these are out there " \
                "is do put your name on it because its the only record I have of actually who was sitting there "

NER_DEMO_SENTENCE = "We need to know by the fifth of April, otherwise there's no need to visit Australia"

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
            []
        ]
    }
]


class TestCalcWordStats(TestCase):
    def test_match_condition_yes_true(self):
        res = filter_action_items(ActionItemState.YES, "yes")
        self.assertTrue(res)

    def test_match_condition_yes_false(self):
        res = filter_action_items(ActionItemState.YES, "no")
        self.assertFalse(res)

    def test_match_condition_no_true(self):
        res = filter_action_items(ActionItemState.NO, "no")
        self.assertTrue(res)

    def test_match_condition_no_false(self):
        res = filter_action_items(ActionItemState.NO, "yes")
        self.assertFalse(res)

    def test_match_condition_all_yes(self):
        res = filter_action_items(ActionItemState.ALL, "yes")
        self.assertTrue(res)

    def test_match_condition_all_no(self):
        res = filter_action_items(ActionItemState.ALL, "no")
        self.assertTrue(res)

    def test_generate_coordinates(self):
        X, y = extract_sorted_xy_coordinates(WORD_COUNT_DICT)
        print(X, y)

    def test_plot_token_counts(self):
        plot_token_counts(WORD_COUNT_DICT, "foo")

    def test_plot_pos_counts(self):
        plot_pos_counts([DEMO_FEATURES])

    def test_feature_extract(self):
        tokens, entities = feature_extract(NER_DEMO_SENTENCE)
        for token in tokens:
            print(token)
        print(entities)

    def test_fetch_all_tokens(self):
        res = read_json(ROOT_PATH / Path('data/extracted-features/Bed003.json'))
        print(res)

    def test_merge_features_from_utterances(self):
        res = extract_features_by_action_state(DEMO_UTTERANCES, ActionItemState.YES)
        self.assertListEqual([[[{'text': 'nice', 'lemma': 'nice', 'pos': 'ADJ', 'isStop': False}], []]], res)

    def test_merge_features_from_utterances_(self):
        res = extract_features_by_action_state(DEMO_UTTERANCES, ActionItemState.ALL)
        print(res)

    def test_map_to_pos(self):
        tokens, entities = feature_extract(DEMO_SENTENCE)
        res = map_to_pos(tokens)
        print(res)

    def test_map_to_text(self):
        tokens, entities = feature_extract(NER_DEMO_SENTENCE)
        res = map_to_text(tokens)
        print(res)

    def test_map_to_entities(self):
        tokens, entities = feature_extract(NER_DEMO_SENTENCE)
        res = map_to_entity(entities)
        print(res)

    def test_plot_token_counts_(self):
        plot_token_counts(WORD_COUNT_DICT, "foo")
