import json
import os
from json import JSONDecodeError
from typing import Dict


def load_json(path: str) -> Dict:
    with open(path, 'r') as file:
        try:
            data = json.load(file)
        except JSONDecodeError as e:
            print(f"Error in file {path}: {e}")
            raise e
    return data


def dump_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
