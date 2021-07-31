import json
from pathlib import Path

from src.base_logger import logger

AIMU_FOLDER = Path("data/aimu.v1.0")
ANNOTATED_ICSI_FOLDER = Path("data/VT2-annotations")
TARGET_FOLDER = Path("data/action-dataset")
FEATURES_FOLDER = Path("data/extracted-features")
MERGED_UTTERANCES_PATH = Path('data/merged_utterances.json')
UTTERANCE_RATIOS_PATH = Path('ratios.md')


def write_file(to_folder: Path, file_path: Path, json_file: str, trgt_suffix) -> None:
    file_path = to_folder / f'{file_path.stem}.{trgt_suffix}'
    with open(file_path, 'w') as file:
        file.write(json_file)


def create_parse_json(entry_parser: callable, strategy: str) -> callable:
    strategy_fun = {
        "file": lambda in_entry: json.load(in_entry),
        "line": lambda in_entry: in_entry
    }[strategy]

    def parse_strategy(file):
        return [entry_parser(line) for line in strategy_fun(file)]

    def parse_json(file_path: Path) -> str:
        with open(file_path) as file:
            return json.dumps(parse_strategy(file))

    return parse_json


def read_json(file: Path):
    with open(file) as f:
        transcript = json.load(f)
    return transcript


def write_to_disk(file: str, trgt_path: Path, suffix: str):
    with open(f'{trgt_path}.{suffix}', 'w') as f:
        f.write(file)
