import json

from detectron2.data import MetadataCatalog
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import Instances
from detectron2.utils.file_io import PathManager


def count_coco_images(json_file: str):
    json_file = PathManager.get_local_path(json_file)
    with open(json_file, "r") as f:
        content = json.load(f)
        return len(content["images"])


def convert_instances_to_coco(dataset_name: str, image_id: int, instances: Instances):
    coco_instances = instances_to_coco_json(instances.to("cpu"), image_id)

    metadata = MetadataCatalog.get(dataset_name)
    if not hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        return coco_instances

    # copied from detectron2.evaluation.coco_evaluation
    # unmap the category ids for COCO
    dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id
    all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
    num_classes = len(all_contiguous_ids)
    assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

    reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
    for result in coco_instances:
        category_id = result["category_id"]
        assert category_id < num_classes, (
            f"A prediction has class={category_id}, "
            f"but the dataset only has {num_classes} classes and "
            f"predicted class id should be in [0, {num_classes - 1}]."
        )
        result["category_id"] = reverse_id_mapping[category_id]
    return coco_instances
