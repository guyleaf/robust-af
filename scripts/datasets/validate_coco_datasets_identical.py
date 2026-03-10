import argparse
from pathlib import Path

from pycocotools.coco import COCO

ANNOTATION_SEARCH_PATTERN = "*.json"


def validate_annotations(root_dir_1: str, root_dir_2: str):
    annotation_dir_1: Path = Path(root_dir_1) / "annotations"
    annotation_dir_2: Path = Path(root_dir_2) / "annotations"

    annotation_files_1 = annotation_dir_1.rglob(ANNOTATION_SEARCH_PATTERN)
    annotation_files_2 = annotation_dir_2.rglob(ANNOTATION_SEARCH_PATTERN)

    annotation_set_1 = set(
        ann_file.relative_to(annotation_dir_1) for ann_file in annotation_files_1
    )
    annotation_set_2 = set(
        ann_file.relative_to(annotation_dir_2) for ann_file in annotation_files_2
    )
    annotation_set = annotation_set_1 & annotation_set_2

    for annotation_file in sorted(annotation_set):
        annotation_file_1 = annotation_dir_1 / annotation_file
        annotation_file_2 = annotation_dir_2 / annotation_file

        print(f"Compared annotations: {annotation_file_1}, {annotation_file_2}")
        assert annotation_file_1.relative_to(
            annotation_dir_1
        ) == annotation_file_2.relative_to(annotation_dir_2), (
            "The path of annotation files are not identical."
        )

        coco_1 = COCO(annotation_file_1)
        coco_2 = COCO(annotation_file_2)

        if coco_1.imgs != coco_2.imgs:
            print(
                f"The images are not identical, {len(coco_1.imgs)}, {len(coco_2.imgs)}."
            )
            for a, b in zip(coco_1.imgs.values(), coco_2.imgs.values()):
                if a != b:
                    print("First non-identical entry:", a, b)
                    break
        if coco_1.anns != coco_2.anns:
            print(
                f"The annotations are not identical, {len(coco_1.anns)}, {len(coco_2.anns)}."
            )
            for a, b in zip(coco_1.anns.values(), coco_2.anns.values()):
                if a != b:
                    print("First non-identical entry:", a, b)
                    break
        print("=================")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate if both datasets are identical (only check common properties)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_dir_1", type=str, help="Path of the root folder")
    parser.add_argument("root_dir_2", type=str, help="Path of the root folder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    validate_annotations(args.root_dir_1, args.root_dir_2)
