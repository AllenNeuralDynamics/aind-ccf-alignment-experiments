#!/usr/bin/env python3

import itk


def compute_distance_map(label_image: itk.Image, mask_value: int) -> itk.Image:
    """Compute the signed distance map image such that each pixel intensity
    represents the estimated shortest distance from the pixel's spatial
    location to the boundary of the masked label region"""
    binary_image = itk.binary_threshold_image_filter(
        label_image, upper_threshold=mask_value, lower_threshold=mask_value
    )

    # Distance map requires floating point input
    if itk.template(binary_image)[1][0] != itk.F:
        binary_image = itk.cast_image_filter(
            binary_image, ttype=[type(binary_image), itk.Image[itk.F, 3]]
        )

    distance_map = itk.approximate_signed_distance_map_image_filter(
        binary_image, inside_value=1, outside_value=0
    )

    return distance_map


def probe_image(input_image: itk.Image, input_points: itk.PointSet) -> list:
    """Get the corresponding voxel intensity value at each point in space"""
    # Isotropic spacing is required so that pixel distances can be
    # mapped to spatial distances for output
    image_spacing = itk.spacing(input_image)[0]
    assert all(
        [
            itk.spacing(input_image)[dim] == image_spacing
            for dim in range(input_image.GetImageDimension())
        ]
    )

    probed_vals = list()

    for point_id in range(input_points.GetNumberOfPoints()):
        point_loc = input_points.GetPoint(point_id)
        image_loc = input_image.TransformPhysicalPointToIndex(point_loc)

        # Behavior is undefined for points that lie outside image extent
        if any(
            [
                image_loc[dim] < itk.origin(input_image)[dim]
                or image_loc[dim]
                > itk.origin(input_image)[dim] + itk.size(input_image)[dim]
                for dim in range(input_image.GetImageDimension())
            ]
        ):
            raise IndexError(
                f"Point {point_id} maps to image index {image_loc} which is outside of the valid image region!"
            )

        probed_val = input_image.GetPixel(image_loc) * image_spacing
        probed_vals.append(probed_val)

    return probed_vals


def describe_list(point_distances):
    """Print a short statistical summary of list of one-dimensional values"""
    import pandas as pd

    df = pd.DataFrame(point_distances)
    print(df.describe())
