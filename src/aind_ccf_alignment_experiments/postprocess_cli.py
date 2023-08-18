#!/usr/bin/env python3

"""
Lightweight CLI application to generate boundary surface from
label map images, upload to S3, and generate surface viewer links.

Accepts batch of label map image inputs as list or glob pattern.
Use `python label_map_to_mesh.py --help` for full parameter list.
"""

import argparse
import glob
import os

import itk
import numpy as np

from .url import get_s3_bucket_path, get_itk_vtk_viewer_link
from .label_methods import (
    generate_mesh_from_label_map,
    get_unique_labels,
    get_label_distance_images,
)
from .point_distance import read_write_probe_image

_yes_to_all = False


def main():
    CCF_BASELINE_LABEL_MESH_BUCKET_PATH = (
        "s3://aind-kitware-collab"
        "/SmartSPIM_652506_2023-01-09_10-18-12_stitched_2023-01-13_19-00-54"
        "/annotations/652506_CCF_Baseline.vtp"
    )

    parser = argparse.ArgumentParser(
        description="Generate mesh surface geometry from label image"
    )
    parser.add_argument(
        "-i",
        "--images",
        nargs="+",
        type=str,
        help="input label image filepath",
    )
    parser.add_argument(
        "-g",
        "--images-pattern",
        type=str,
        help="Glob pattern for images to process",
    )
    parser.add_argument(
        "--mesh-path", type=str, help="Path to directory for mesh output"
    )
    parser.add_argument(
        "--aws-root-bucket", type=str, help="Root AWS S3 bucket"
    )
    parser.add_argument(
        "--aws-subject-bucket",
        type=str,
        help="Subject bucket for AWS S3 upload",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Experiment bucket name or path for AWS S3 upload",
    )
    parser.add_argument(
        "--split-labels",
        action="store_true",
        help="Generate one mesh per label instead of a single, overarching binary mesh",
    )
    parser.add_argument(
        "--probe-images",
        action="store_true",
        help="Sample voxel intensity values at mesh vertices"
        'and store as mesh point data. Requires that "split-labels" is turned on.',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate output filepaths and commands without actually processing data",
    )
    parser.add_argument(
        "-y",
        "--yes-to-all",
        action="store_true",
        help="Automatically respond 'yes' to each prompt",
    )
    args = parser.parse_args()

    if args.yes_to_all:
        _yes_to_all = True

    if args.images:
        image_filepaths = args.images
    elif args.images_pattern:
        image_filepaths = glob.glob(args.images_pattern)

    print(f"Images: {image_filepaths}")
    if not _yes_to_all:
        input("Press any key to continue...")

    mesh_filepaths = []

    if args.probe_images and not args.dry_run:
        # TODO generalize to other distance map paths
        assert all(
            ["652506" in image_filepath for image_filepath in image_filepaths]
        )
        distance_images = get_label_distance_images(
            distance_images_path=r"D:\repos\allen-registration\notebooks\data\input\652506\annotation\distances"
        )

    # TODO update for general AWS S3 case for mesh baseline
    PUT_AWS_BUCKETS = args.aws_root_bucket and args.aws_subject_bucket
    if args.aws_root_bucket:
        assert args.aws_root_bucket == "aind-kitware-collab"
    if args.aws_subject_bucket:
        assert (
            args.aws_subject_bucket
            == "SmartSPIM_652506_2023-01-09_10-18-12_stitched_2023-01-13_19-00-54"
        )

    mesh_bucket_paths = []
    for image_filepath in image_filepaths:
        image_mesh_bucket_paths = []
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"Processing {image_filepath}")

        mesh_path = (
            args.mesh_path
            if args.mesh_path
            else os.path.dirname(image_filepath)
        )
        mesh_filepaths = []

        if args.split_labels:
            label_values = get_unique_labels(image_filepath)
            for label_value in label_values:
                if image_filepath.endswith(".nii.gz"):
                    mesh_filepath = f'{mesh_path}/{os.path.basename(image_filepath)[:-len(".nii.gz")]}_{label_value}.vtp'
                else:
                    mesh_filepath = f"{mesh_path}/{os.path.splitext(os.path.basename(image_filepath))[0]}_{label_value}.vtp"

                if not args.dry_run:
                    _ = generate_mesh_from_label_map(
                        image_filepath,
                        mesh_filepath=mesh_filepath,
                        label_value=label_value,
                        dry_run=args.dry_run,
                    )

                    if args.probe_images:
                        _ = read_write_probe_image(
                            distance_image=distance_images[label_value],
                            mesh_filepath=mesh_filepath,
                        )

                mesh_filepaths.append(mesh_filepath)
                print(f"Mesh written to {mesh_filepath}")
        else:
            mesh_filepath = (
                f"{mesh_path}/{os.path.splitext(image_filepath)[0]}.vtp"
            )

            if not args.dry_run:
                output_mesh_filepath = generate_mesh_from_label_map(
                    image_filepath,
                    mesh_filepath=mesh_filepath,
                    dry_run=args.dry_run,
                )
            mesh_filepaths = [output_mesh_filepath]
            print(f"Mesh written to {output_mesh_filepath}")

        if PUT_AWS_BUCKETS:
            for mesh_filepath in mesh_filepaths:
                bucket_path = get_s3_bucket_path(
                    root_name=args.aws_root_bucket,
                    subject_s3_dir=args.aws_subject_bucket,
                    sample_filepath=mesh_filepath,
                    registration_experiment_id=args.experiment_id,
                )
                image_mesh_bucket_paths.append(bucket_path)
                mesh_bucket_paths.append(bucket_path)

                s3_upload_command = (
                    f'aws s3 cp "{mesh_filepath}" "{bucket_path}"'
                )

                print(f"\nAWS S3 upload command: {s3_upload_command}\n")

                if not args.dry_run:
                    allow_upload = (
                        "Y"
                        if _yes_to_all
                        else input("Proceed with upload? (Y/N/E/A) >> ")
                    )
                    if allow_upload == "Y":
                        print(f"Executing upload command...")
                        os.system(s3_upload_command)
                    elif allow_upload == "A":
                        print(f"Uploading all.")
                        _yes_to_all = True
                        os.system(s3_upload_command)
                    elif allow_upload == "E":
                        print(f"Aborting.\n")
                        exit(1)
                    else:
                        print(f"Skipped.\n")
                        continue

                individual_viewer_link = get_itk_vtk_viewer_link(
                    [CCF_BASELINE_LABEL_MESH_BUCKET_PATH, bucket_path]
                )
                print(f"Mesh label viewer link: {individual_viewer_link}")

            if args.split_labels:
                individual_viewer_link = get_itk_vtk_viewer_link(
                    [
                        CCF_BASELINE_LABEL_MESH_BUCKET_PATH,
                        *image_mesh_bucket_paths,
                    ]
                )
                print(
                    f"Mesh viewer link for {image_filepath}: {individual_viewer_link}"
                )

    if PUT_AWS_BUCKETS:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        combined_viewer_link = get_itk_vtk_viewer_link(
            [CCF_BASELINE_LABEL_MESH_BUCKET_PATH, *mesh_bucket_paths]
        )
        print(f"Combined mesh label viewer link: {combined_viewer_link}")


if __name__ == "__main__":
    main()
