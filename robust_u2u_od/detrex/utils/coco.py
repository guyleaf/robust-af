import json

from detectron2.utils.file_io import PathManager


def count_coco_images(json_file: str):
    json_file = PathManager.get_local_path(json_file)
    with open(json_file, "r") as f:
        content = json.load(f)
        return len(content["images"])
