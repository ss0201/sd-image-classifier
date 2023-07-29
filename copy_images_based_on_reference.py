import argparse
import logging
import multiprocessing
import os
import shutil
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageChops, ImageOps

UNMATCHED_DIR_NAME = "_unmatched"


@lru_cache(maxsize=None)
def process_image(file: str) -> Image.Image:
    im = Image.open(file)
    im = ImageOps.grayscale(im)
    return im


def calc_image_similarity(file1: str, file2: str) -> float:
    im1 = process_image(file1)
    im2 = process_image(file2)
    diff_histogram = np.array(ImageChops.difference(im1, im2).histogram())

    zero_diff_ratio = diff_histogram[0] / np.sum(diff_histogram)
    return zero_diff_ratio


def build_file_cache(dirs: list[str]) -> dict[str, set[str]]:
    cache: dict[str, set[str]] = {}
    for dir in dirs:
        if os.path.isdir(dir):
            cache[dir] = set(os.listdir(dir))
    return cache


def find_matching_reference_file(
    file_path: str,
    reference_dirs: list[str],
    similarity_threshold: float,
    file_cache: dict[str, set[str]],
) -> tuple[Optional[str], Optional[str], Optional[float]]:
    # First check if there is a file with the same name in the reference dirs
    # so we can avoid calculating the similarity for all files in the reference dirs
    basename = os.path.basename(file_path)
    for reference_dir in reference_dirs:
        if basename in file_cache[reference_dir]:
            reference_file_path = os.path.join(reference_dir, basename)
            similarity = calc_image_similarity(file_path, reference_file_path)
            if similarity > similarity_threshold:
                return reference_dir, basename, similarity

    # Otherwise, check the similarity of all files in the reference dirs
    for reference_dir in reference_dirs:
        for reference_file in file_cache[reference_dir]:
            reference_file_path = os.path.join(reference_dir, reference_file)
            similarity = calc_image_similarity(file_path, reference_file_path)
            if similarity > similarity_threshold:
                return reference_dir, reference_file, similarity

    return None, None, None


def process_file(
    src_file: str,
    work_dir: str,
    src_dir: str,
    reference_dirs: list[str],
    similarity_threshold: float,
    file_cache: dict[str, set[str]],
    processed_dirs: list[str],
) -> None:
    if any(
        src_file in file_cache[processed_dir]
        for processed_dir in processed_dirs
        if processed_dir in file_cache
    ):
        logging.info(f"Skipping {src_file}")
        return

    src_file_path = os.path.join(src_dir, src_file)
    matching_reference_dir, matching_file, similarity = find_matching_reference_file(
        src_file_path, reference_dirs, similarity_threshold, file_cache
    )
    dest_dir_name = (
        UNMATCHED_DIR_NAME
        if matching_reference_dir is None
        else os.path.basename(matching_reference_dir)
    )
    dest_path = os.path.join(work_dir, dest_dir_name, src_file)

    logging.info(
        f"Copying {src_file}\n"
        f"   -> {dest_dir_name} (similarity: {similarity})\n"
        f"   Matching file: {matching_file}"
    )

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(src_file_path, dest_path)


def copy_src_files_to_work_dir_based_on_reference(
    work_dir: str, src_dir: str, reference_dirs: list[str], similarity_threshold: float
) -> None:
    processed_dirs = [
        os.path.join(work_dir, dir_name)
        for dir_name in [os.path.basename(ref_dir) for ref_dir in reference_dirs]
        + [UNMATCHED_DIR_NAME]
    ]

    file_cache = build_file_cache([src_dir] + processed_dirs + reference_dirs)

    with multiprocessing.Pool() as pool:
        pool.starmap(
            process_file,
            [
                (
                    src_file,
                    work_dir,
                    src_dir,
                    reference_dirs,
                    similarity_threshold,
                    file_cache,
                    processed_dirs,
                )
                for src_file in os.listdir(src_dir)
            ],
        )


def main():
    parser = argparse.ArgumentParser(
        description="Copy source images to working directories based on "
        "similarity with reference images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dirs_group = parser.add_argument_group("dirs")
    dirs_group.add_argument(
        "--work-dir",
        type=str,
        help="Working directory to copy the images to.",
        required=True,
    )
    dirs_group.add_argument(
        "--src-dir",
        type=str,
        help="Directory containing the source images.",
        required=True,
    )
    dirs_group.add_argument(
        "--ref-dirs",
        type=str,
        nargs="+",
        help="Directories containing the reference images.",
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for images (0-1).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Copying files from {args.src_dir} to {args.work_dir}...")

    copy_src_files_to_work_dir_based_on_reference(
        args.work_dir,
        args.src_dir,
        args.ref_dirs,
        args.threshold,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
