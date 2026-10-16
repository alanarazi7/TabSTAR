import json
import os
from json import JSONDecodeError
from typing import Dict, List, Any
from zipfile import ZipFile


def load_json(path: str) -> Dict:
    with open(path, 'r') as file:
        try:
            data = json.load(file)
        except JSONDecodeError as e:
            print(f"Error in file {path}: {e}")
            raise e
    return data

def load_json_lines(path: str) -> List[Dict]:
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def dump_json(data: Dict, path: str) -> None:
    create_dir(path)
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def dump_json_lines(data, path: str) -> None:
    with open(path, 'w') as file:
        for d in data:
            json.dump(d, file)
            file.write('\n')

def load_txt(path: str) -> str:
    with open(path, 'r') as file:
        data = file.read()
    return data

def dump_txt(data: str, path: str) -> None:
    create_dir(path)
    with open(path, 'w') as file:
        file.write(data)


def create_dir(path: str):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def unzip_file(zip_path: str):
    assert zip_path.endswith('.zip')
    extract_to = zip_path.rstrip('.zip')
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def is_img_dict(value: Any) -> bool:
    return isinstance(value, dict) and 'bytes' in value