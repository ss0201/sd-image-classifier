import argparse
import logging
import operator
import os
import shutil
from functools import reduce
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

UNMATCHED_DIR_NAME = "_unmatched"


def calc_image_similarity(file1, file2):
    im1 = Image.open(file1)
    im2 = Image.open(file2)
    im1 = ImageOps.grayscale(im1)
    im2 = ImageOps.grayscale(im2)
    histogram = ImageChops.difference(im1, im2).histogram()

    zero_diff_ratio = histogram[0] / reduce(operator.add, histogram)
    return zero_diff_ratio


def find_matching_train_dir(file_path, train_dirs, similarity_threshold):
    for train_dir in train_dirs:
        train_file_path = os.path.join(train_dir, os.path.basename(file_path))
        if os.path.exists(train_file_path):
            similarity = calc_image_similarity(file_path, train_file_path)
            if similarity > similarity_threshold:
                logging.info(
                    f"Found matching file {train_file_path} (similarity: {similarity})"
                )
                return train_dir

    for train_dir in train_dirs:
        for train_file in os.listdir(train_dir):
            train_file_path = os.path.join(train_dir, train_file)
            similarity = calc_image_similarity(file_path, train_file_path)
            if similarity > similarity_threshold:
                logging.info(
                    f"Found matching file {train_file_path} (similarity: {similarity})"
                )
                return train_dir
    return None


def copy_src_files_to_work_dir(work_dir, src_dir, train_dirs, similarity_threshold):
    for src_file in os.listdir(src_dir):
        if any(
            os.path.exists(
                os.path.join(work_dir, os.path.basename(train_dir), src_file)
            )
            for train_dir in train_dirs + [UNMATCHED_DIR_NAME]
        ):
            logging.info(f"Skipping {src_file}...")
            continue

        logging.info(f"Processing {src_file}...")

        src_file_path = os.path.join(src_dir, src_file)
        matching_train_dir = find_matching_train_dir(
            src_file_path, train_dirs, similarity_threshold
        )
        dest_dir_name = (
            UNMATCHED_DIR_NAME
            if matching_train_dir is None
            else os.path.basename(matching_train_dir)
        )
        dest_path = os.path.join(work_dir, dest_dir_name, src_file)

        logging.info(f"Copying {src_file_path} to {dest_path}...")

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_file_path, dest_path)


def main():
    parser = argparse.ArgumentParser(
        description="Copy source images to training directories based on similarity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dirs_group = parser.add_argument_group("dirs")
    dirs_group.add_argument(
        "--work_dir",
        type=str,
        help="Working directory to copy the images to.",
        required=True,
    )
    dirs_group.add_argument(
        "--src_dir",
        type=str,
        help="Directory containing the source images.",
        required=True,
    )
    dirs_group.add_argument(
        "--train_dirs",
        type=str,
        nargs="+",
        help="Directories containing the training images.",
        required=True,
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for images (0-1).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())
    logging.info(f"Copying files from {args.src_dir} to {args.work_dir}...")

    copy_src_files_to_work_dir(
        Path(args.work_dir), Path(args.src_dir), args.train_dirs, args.threshold
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()