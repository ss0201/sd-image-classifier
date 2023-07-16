import argparse
import logging
import operator
import os
import shutil
from functools import reduce
from pathlib import Path

from PIL import Image, ImageChops, ImageOps


def calc_image_similarity(file1, file2):
    im1 = Image.open(file1)
    im2 = Image.open(file2)
    im1 = ImageOps.grayscale(im1)
    im2 = ImageOps.grayscale(im2)
    histogram = ImageChops.difference(im1, im2).histogram()

    zero_diff_ratio = histogram[0] / reduce(operator.add, histogram)
    return zero_diff_ratio


def find_matching_train_dir(file_path, train_dirs):
    for train_dir in train_dirs:
        train_file_path = os.path.join(train_dir, os.path.basename(file_path))
        if os.path.exists(train_file_path):
            similarity = calc_image_similarity(file_path, train_file_path)
            if similarity > 0.5:
                return train_dir

    for train_dir in train_dirs:
        for train_file in os.listdir(train_dir):
            train_file_path = os.path.join(train_dir, train_file)
            similarity = calc_image_similarity(file_path, train_file_path)
            if similarity > 0.5:
                return train_dir
    return None


def copy_src_files_to_work_dir(work_dir, src_dir, train_dirs):
    for file in os.listdir(src_dir):
        logging.info(f"Processing {file}...")

        file_path = os.path.join(src_dir, file)
        matching_train_dir = find_matching_train_dir(file_path, train_dirs)
        dest_dir_name = (
            "_unmatched"
            if matching_train_dir is None
            else os.path.basename(matching_train_dir)
        )
        dest_path = os.path.join(work_dir, dest_dir_name, file)

        logging.info(f"Copying {file_path} to {dest_path}...")

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(file_path, dest_path)


def main():
    parser = argparse.ArgumentParser(description="Setup images for training.")
    dirs_group = parser.add_argument_group("dirs")
    dirs_group.add_argument(
        "--work_dir",
        type=str,
        default="work_dir",
        help="Working directory.",
        required=True,
    )
    dirs_group.add_argument(
        "--src_dir", type=str, default="all", help="Source directory.", required=True
    )
    dirs_group.add_argument(
        "--train_dirs", type=str, nargs="+", help="Training directories.", required=True
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level.")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())
    logging.info(f"Copying files from {args.src_dir} to {args.work_dir}...")

    copy_src_files_to_work_dir(Path(args.work_dir), Path(args.src_dir), args.train_dirs)

    logging.info("Done.")


if __name__ == "__main__":
    main()
