#!/usr/bin/env python3

"""
Helpers for postprocessing and evaluating voxel label map image annotations
"""

import os
from typing import List, Tuple, Dict

import itk
import pandas as pd


def compose_label_image(
    labels_info: pd.DataFrame,
    input_labels_path: str,
    output_path: str = None,
    verbose: bool = False,
) -> itk.Image:
    """
    Compose several expert segmentations into a single label image file.
    Assumptions:
    1. Each input file is a binary image representing a single label
    2. Input files correspond in image and physical space (same metadata and voxel size)
    3. There are no voxel overlaps between label regions
    """

    composed_label_image = None
    for index, row in labels_info.iterrows():
        label_image_filepath = f'{input_labels_path}/{row["seg_name"]}.nii.gz'
        if verbose:
            print(f"read from {label_image_filepath}")
        source_label_image = itk.imread(label_image_filepath)

        source_label_image = itk.cast_image_filter(
            source_label_image,
            ttype=[type(source_label_image), itk.Image[itk.F, 3]],
        )
        source_label_image = itk.binary_threshold_image_filter(
            source_label_image,
            lower_threshold=1,
            inside_value=int(row["ccf_label_value"]),
        )

        if not composed_label_image:
            composed_label_image = source_label_image
        else:
            add_filter = itk.AddImageFilter[
                type(composed_label_image),
                type(composed_label_image),
                type(composed_label_image),
            ].New()
            for index, image in enumerate(
                [composed_label_image, source_label_image]
            ):
                add_filter.SetInput(index, image)
            add_filter.Update()
            composed_label_image = add_filter.GetOutput()
            del source_label_image

    if output_path:
        itk.imwrite(composed_label_image, output_path, compression=True)
        if verbose:
            print(f"Composed label image written to {output_path}")

    return composed_label_image


def compute_dice_coefficient(
    source_image: itk.Image, target_image: itk.Image
) -> Tuple[float, itk.LabelOverlapMeasuresImageFilter]:
    """Compute the dice coefficient to compare volume overlap between two label regions"""
    dice_filter = itk.LabelOverlapMeasuresImageFilter[type(source_image)].New()
    dice_filter.SetInput(source_image)
    dice_filter.SetTargetImage(target_image)
    dice_filter.Update()
    return dice_filter.GetDiceCoefficient(), dice_filter


def convert_mesh_to_extension(input_filepath: str, output_filepath: str):
    """Use VTK to convert a mesh from one supported file type to another."""
    import vtk

    FILE_EXTENSION_STR_LEN = 3

    supported_readers = {
        "obj": vtk.vtkOBJReader,
        "vtk": vtk.vtkPolyDataReader,
        "vtp": vtk.vtkXMLPolyDataReader,
    }

    if not any(
        [
            input_filepath[-FILE_EXTENSION_STR_LEN:] == ext
            for ext in supported_readers
        ]
    ):
        raise ValueError(
            f"Could not read from file extension for {input_filepath}"
        )
    if not output_filepath.endswith(".vtp"):
        raise ValueError(
            f"Only write to XML PolyData file (.vtp) is supported at this time,"
            "received {output_filepath}"
        )

    reader = supported_readers[input_filepath[-FILE_EXTENSION_STR_LEN:]]()
    reader.SetFileName(input_filepath)
    reader.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(reader.GetOutput())
    writer.SetFileName(output_filepath)
    writer.Update()


def generate_mesh_from_label_map(
    image_filepath,
    mesh_filepath: str = None,
    label_value: int = None,
    dry_run: bool = False,
) -> str:
    """
    Generate a surface from a label image and write out in VTK XML PolyData format
    to the input directory
    """
    BACKGROUND_VALUE = 0
    BINARY_OBJECT_VALUE = 1
    OUTPUT_EXTENSION = "vtp"

    upper_threshold = label_value if label_value else 65535
    lower_threshold = label_value if label_value else 1

    if image_filepath.endswith(".nii.gz"):
        input_basename = image_filepath.replace(".nii.gz", "")
    else:
        input_basename = os.path.basename(image_filepath)

    if not mesh_filepath:
        mesh_filepath = f"{input_basename}.{OUTPUT_EXTENSION}"
    else:
        os.makedirs(os.path.dirname(mesh_filepath), exist_ok=True)
    if not mesh_filepath.endswith(OUTPUT_EXTENSION):
        raise ValueError(
            f"Mesh output must have extension .{OUTPUT_EXTENSION}"
        )

    tmp_mesh_filepath = f"{input_basename}.obj"

    if dry_run:
        return mesh_filepath

    assert os.path.exists(image_filepath)
    label_image = itk.imread(image_filepath)

    binary_label_image = itk.binary_threshold_image_filter(
        label_image,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        inside_value=BINARY_OBJECT_VALUE,
    )
    label_mesh = itk.binary_mask3_d_mesh_source(
        binary_label_image, object_value=BINARY_OBJECT_VALUE
    )

    itk.meshwrite(label_mesh, tmp_mesh_filepath, compression=True)
    convert_mesh_to_extension(tmp_mesh_filepath, mesh_filepath)
    return mesh_filepath


def get_unique_labels(image_filepath: str) -> list:
    """
    Load a label image and get a list of unique labels in the image
    """
    BACKGROUND_VALUE = 0

    import numpy as np

    print(f"Reading label image to get unique labels...")
    label_image = itk.imread(image_filepath)
    label_values = list(np.unique(label_image))
    label_values.remove(BACKGROUND_VALUE)
    return label_values


def get_label_distance_images(
    distance_images_path: str,
) -> Dict[int, itk.Image[itk.F, 3]]:
    """
    Load signed label distance maps from disk and return as a dictionary
    mapping from a label value to its distance map
    """
    import re
    import glob

    label_distance_images = {}

    distance_image_filepaths = glob.glob(
        f"{distance_images_path}/distance_*.nii.gz"
    )

    for distance_image_filepath in distance_image_filepaths:
        label_value = int(
            re.match(
                ".*distance_([0-9]*).nii.gz", distance_image_filepath
            ).group(1)
        )
        print(f"Reading distance image for label value {label_value}")
        distance_image = itk.imread(distance_image_filepath)
        label_distance_images[label_value] = distance_image

    return label_distance_images
