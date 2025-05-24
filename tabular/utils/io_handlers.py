import json
from json import JSONDecodeError
from typing import Dict, List


def load_json(path: str) -> Dict:
    with open(path, 'r') as file:
        try:
            data = json.load(file)
        except JSONDecodeError as e:
            print(f"Error in file {path}: {e}")
            raise e
    return data


def dump_json(data: Dict, path: str) -> None:
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
