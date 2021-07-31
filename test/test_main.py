from datetime import date
from pathlib import Path
from unittest import TestCase

from src.base_logger import ROOT_PATH
from src.main import create_argparse, main, Task, convert_and_store
from src.process_aimu import remove_tags, remove_non_alphanum, is_action_item
from src.utils import ANNOTATED_ICSI_FOLDER, FEATURES_FOLDER, create_parse_json, write_to_disk
from src.word_stats import parse_icsi_json, plot_all_token_counts, format_title, Feature, ActionItemState

UTTERANCE_STRING_NOT_ACTION = "Bed010	1767.342	1775.030	mn015	<search> It's <query_term> KLEIST" \
                              "</query_term> </search>. It's the uh <query_term> Bielefeld generation of uh spatial" \
                              "descriptions </query_term> and whatever. "
UTTERANCE_STRING_IS_ACTION = "Bed010	1776.430	1778.950	me010	<send_email> Well, that may be another thing " \
                             "that <contact_name> Keith </contact_name> wants to <email_content> look at " \
                             "</email_content> </send_email>. "


class TestMain(TestCase):
    def test_create_argparse(self):
        parser = create_argparse()
        args = parser.parse_args(["--task", "format-aimu"])

        self.assertEqual("format-aimu", str(args.task))

    def test_main(self):
        main(Task.FORMAT_AIMU)

    def test_is_action_item_no(self):
        res = is_action_item(UTTERANCE_STRING_NOT_ACTION)
        self.assertEqual(res, "no")

    def test_is_action_item_yes(self):
        res = is_action_item(UTTERANCE_STRING_IS_ACTION)
        self.assertEqual(res, "yes")

    def test_text_remove_tags(self):
        res = remove_tags("<one>first text </one><two_tags>second text</two_tags> and some more text")
        self.assertEqual(res, "first text second text and some more text")

    def test_remove_non_alphanum(self):
        res = remove_non_alphanum("\"I'd do it    in a- a- split sec...\" she    said _file_strange")
        self.assertEqual(res, "Id do it in a a split sec she said filestrange")

    def test_parse_single_file(self):
        parse_json = create_parse_json(ROOT_PATH / "Bro003.json")
        res = parse_json(Path("../data/aimu.v1.0/Bed003.trans"))
        print(type(res))

    def test_convert_and_store(self):
        convert_and_store(src_folder=ROOT_PATH / ANNOTATED_ICSI_FOLDER,
                          trgt_folder=ROOT_PATH / FEATURES_FOLDER,
                          src_suffix='*.json',
                          trgt_suffix="json",
                          file_parser=create_parse_json()
                          )

    def test_create_parse_json(self):
        parse_json = create_parse_json(parse_icsi_json, "file")
        res = parse_json(ROOT_PATH / Path("data/manually-annotated/Bro003.json"))
        print(res)

    def test_format_title(self):
        res = format_title(Feature.TEXT, ActionItemState.YES)
        print(res)

    def test_plot_all_token_counts(self):
        plot_all_token_counts(ROOT_PATH / Path(FEATURES_FOLDER))

    def test_write_to_disk(self):
        write_to_disk("I'm a test",
                      ROOT_PATH / Path("data") / f'{date.today()}-evaluation',
                      "md")
