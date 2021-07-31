import argparse
import os
import sys
from enum import Enum
from pathlib import Path

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_PATH)
from src.base_logger import logger
from src.word_stats import parse_icsi_json, plot_all_token_counts
from src.process_aimu import parse_line_aimu
from src.utils import write_file, create_parse_json, AIMU_FOLDER, ANNOTATED_ICSI_FOLDER, FEATURES_FOLDER, \
    MERGED_UTTERANCES_PATH, UTTERANCE_RATIOS_PATH
from src.analysis import write_merged_utterances, load_utterances_flat, calculate_utterance_ratios, run_classifier


class Task(Enum):
    FORMAT_AIMU = "format-aimu"
    COUNT_HIST = "count-hist"
    FEATURE_EXTRACT = "feature-extract"
    MERGE_UTTERANCES = "merge-utterances"
    UTTERANCE_RATIO = "utterance-ratio"
    RUN_CLASSIFIER = "run-classifier"

    def __str__(self):
        return self.value


def create_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=Task)
    return parser


def convert_and_store(src_folder: Path, trgt_folder: Path, src_suffix: str, trgt_suffix: str,
                      file_parser: callable) -> None:
    file_list = list(src_folder.glob(src_suffix))
    processed_files = [file_parser(path) for path in file_list]

    for file_path, json_file in zip(file_list, processed_files):
        logger.info(f"Processing file: {file_path}")
        write_file(trgt_folder, file_path, json_file, trgt_suffix)


def write_utterance_ratios(utterances_path: Path, ratios_path: Path):
    utterances = load_utterances_flat(utterances_path)
    calculate_utterance_ratios(utterances, ratios_path)


def main(task: Task) -> None:
    if task == Task.FORMAT_AIMU:
        logger.info("Task: Convert aimu files to json")
        convert_and_store(src_folder=ROOT_PATH / Path(AIMU_FOLDER),
                          trgt_folder=ROOT_PATH / Path(ANNOTATED_ICSI_FOLDER),
                          src_suffix='*.trans',
                          trgt_suffix="json",
                          file_parser=create_parse_json(parse_line_aimu, strategy="line")
                          )
        logger.info("Finished conversion")
    if task == Task.FEATURE_EXTRACT:
        logger.info("Task: Extract features from corpus")
        convert_and_store(src_folder=ROOT_PATH / Path(ANNOTATED_ICSI_FOLDER),
                          trgt_folder=ROOT_PATH / Path(FEATURES_FOLDER),
                          src_suffix='*.json',
                          trgt_suffix="json",
                          file_parser=create_parse_json(parse_icsi_json, strategy="file")
                          )
        logger.info("Finished conversion")
    if task == Task.COUNT_HIST:
        plot_all_token_counts(ROOT_PATH / Path(FEATURES_FOLDER))
    if task == Task.MERGE_UTTERANCES:
        write_merged_utterances(src_folder=ROOT_PATH / Path(FEATURES_FOLDER),
                                trgt_path=ROOT_PATH / Path(MERGED_UTTERANCES_PATH))
    if task == Task.UTTERANCE_RATIO:
        write_utterance_ratios(utterances_path=ROOT_PATH / Path(MERGED_UTTERANCES_PATH),
                               ratios_path=ROOT_PATH / Path('data') / UTTERANCE_RATIOS_PATH)
    if task == Task.RUN_CLASSIFIER:
        run_classifier(ROOT_PATH / Path(MERGED_UTTERANCES_PATH), "stacked-ent")


if __name__ == '__main__':
    parser = create_argparse()
    args = parser.parse_args()
    main(args.task)
