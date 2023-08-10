#!/usr/bin/env python3

"""
Helpers for dealing with data locations and naming schema:
- SmartSPIM filepaths
- AWS S3 buckets
- ITK-VTK-Viewer URLs
"""

import os
import re
from typing import List, Tuple


def parse_sample_filepath(sample_filepath: str) -> Tuple[str, str, str, str]:
    """
    Parse sample filepath parameters based on naming rules.

    Returns subject ID, channel ID, registration date, and experiment name.
    """

    sample_filepath = sample_filepath.replace("\\", "/")

    subject_id = int(
        re.match(".*/results/([0-9]*)/.*", sample_filepath).group(1)
    )
    channel_id = re.match(
        ".*/(Ex_[0-9]*_Em_[0-9]*)/.*", sample_filepath
    ).group(1)
    registration_date = re.match(
        ".*/([0-9]{4}\.[0-9]{2}\.[0-9]{2})/.*", sample_filepath
    ).group(1)
    experiment_name = re.match(
        f".*/{subject_id}/([\w/]*)/{registration_date}/{channel_id}/.*",
        sample_filepath,
    ).group(1)

    return subject_id, channel_id, registration_date, experiment_name


def get_s3_bucket_path(
    root_name: str,
    subject_s3_dir: str,
    sample_filepath: str,
    registration_experiment_id: str = None,
) -> str:
    """
    Generate bucket string for upload to AWS.

    Local string is expected to follow subject/experiment/date/channel format:
    "D:\repos\allen-registration\notebooks\data\results\652506\LEVEL_3\2023.06.05\Ex_488_Em_525\labels\652506_Ex_488_Em_525_labels.nii.gz"

    AWS S3 upstream bucket path is expected to follow subject/date/experiment/channel format:
    s3://aind-kitware-collab/SmartSPIM_652506_2023-01-09_10-18-12_stitched_2023-01-13_19-00-54/registered/2023.06.05/L3/Ex_488_Em_525/labels/652506_Ex_488_Em_525_labels.nii.gz
    """

    subject_id = int(sample_filepath.split("\\results\\")[1].split("\\")[0])
    assert str(subject_id) in subject_s3_dir

    registration_date = re.match(
        ".*([0-9]{4}\.[0-9]{2}\.[0-9]{2}).*", sample_filepath
    ).group(1)
    channel_id = re.match(".*(Ex_[0-9]*_Em_[0-9]*).*", sample_filepath).group(
        1
    )

    if not registration_experiment_id:
        registration_experiment_id = (
            sample_filepath.split(str(subject_id))[1]
            .split(registration_date)[0]
            .strip("\\/")
        )

    is_label = "label" in sample_filepath

    bucket_result = (
        f"s3://{root_name}"
        f"/{subject_s3_dir}/registered"
        f"/{registration_date}/{registration_experiment_id}/{channel_id}"
        f'{"/labels" if is_label else ""}'
        f"/{os.path.basename(sample_filepath)}"
    )

    return bucket_result


def get_s3_https_bucket_data_url(s3_bucket_path: str) -> str:
    """
    Compose HTTPS link from S3 URL
    """
    if not s3_bucket_path.startswith("s3://"):
        raise ValueError(f"Not an AWS S3 path: {s3_bucket_path}")

    bucket_parts = s3_bucket_path.replace("s3://", "").rstrip("/").split("/")
    s3_root_bucket = bucket_parts[0]
    s3_bucket_subpath = "/".join(bucket_parts[1:])
    return f"https://{s3_root_bucket}.s3.amazonaws.com/{s3_bucket_subpath}"


def get_itk_vtk_viewer_link(s3_bucket_paths: List) -> str:
    """
    Generate URL for ITK-VTK-Viewer with given S3 files.
    """
    if type(s3_bucket_paths) is str:
        s3_bucket_paths = [s3_bucket_paths]

    ITK_VTK_VIEWER_URL = (
        "https://kitware.github.io/itk-vtk-viewer/app/?fileToLoad="
    )
    data_urls = [
        get_s3_https_bucket_data_url(data_path)
        for data_path in s3_bucket_paths
    ]

    return f'{ITK_VTK_VIEWER_URL}{",".join(data_urls)}'
