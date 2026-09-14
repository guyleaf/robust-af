import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy checkpoints to the destination folder"
    )
    parser.add_argument("src_dir", help="The source folder")
    parser.add_argument("dst_dir", help="The destination folder")
    parser.add_argument(
        "--pattern",
        default="*best*.pt?",
        help="The search pattern for the file (Unix-style wildcards)",
    )
    parser.add_argument(
        "--force",
        default=False,
        help="Overwrite if the file exists in the destination folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for exp_folder in src_dir.iterdir():
        if not exp_folder.is_dir():
            continue

        files = list(exp_folder.glob(args.pattern))
        if len(files) == 0:
            print(f"{exp_folder}, skipped.")
            continue
        assert len(files) == 1, "Found multiple files in the experiment folder."
        checkpoint = files[0]

        out_dir = dst_dir / exp_folder.name
        out_dir.mkdir(exist_ok=True)

        if args.force or not (out_dir / checkpoint.name).is_file():
            shutil.copy2(checkpoint, out_dir)
        else:
            print(f"{checkpoint} file exists, skipped.")
