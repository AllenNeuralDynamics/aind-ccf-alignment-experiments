#!/usr/bin/env python3

"""
Helpers for evaluating registration results with surface point distance comparisons
"""


import os
from typing import List, Tuple, Union

import itk
import vtk


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


def probe_image_at_point(
    distance_image: itk.Image, point: List[float]
) -> float:
    image_spacing = itk.spacing(distance_image)[0]
    if not all(
        [
            itk.spacing(distance_image)[dim] == image_spacing
            for dim in range(distance_image.GetImageDimension())
        ]
    ):
        raise ValueError("Cannot probe image with anisotropic spacing")

    index = distance_image.TransformPhysicalPointToIndex(point)

    # Behavior is undefined for points that lie outside image extent
    if any(
        [
            index[dim] < 0 or index[dim] > itk.size(distance_image)[dim]
            for dim in range(distance_image.GetImageDimension())
        ]
    ):
        raise IndexError(
            f"Point {point} maps to image index {index} which is outside of the valid image region!"
        )

    # Sample the point at the given voxel without interpolation.
    # Point distances are multiplied by the image spacing to convert
    # 1D distance from voxel space to physical space
    return distance_image.GetPixel(index) * image_spacing


def probe_image(
    input_image: itk.Image, input_mesh: Union[vtk.vtkPolyData,itk.Mesh[itk.F, 3]]
) -> Tuple[List[float], Union[vtk.vtkPolyData,itk.Mesh[itk.F, 3]]]:
    """Get the corresponding voxel intensity value at each point in space"""
    if type(input_mesh) == vtk.vtkPolyData:
        return probe_image_vtk(
            distance_image=input_image, input_mesh=input_mesh
        )
    elif (
        type(input_mesh) == itk.Mesh[itk.F, 3]
        or type(input_mesh) == itk.PointSet[itk.F, 3]
    ):
        return probe_image_itk(
            input_image=input_image, input_points=input_mesh
        )
    else:
        raise TypeError(f"Unsupported input mesh type: {type(input_mesh)}")


def probe_image_itk(
    input_image: itk.Image, input_points: itk.PointSet
) -> List[float]:
    """Get the corresponding voxel intensity value at each point in space"""
    probed_vals = list()

    for point_id in range(input_points.GetNumberOfPoints()):
        point_loc = input_points.GetPoint(point_id)
        probed_vals.append(probe_image_at_point(input_image, point_loc))

    return probed_vals


def probe_image_vtk(
    distance_image: itk.Image,
    input_mesh: vtk.vtkPolyData,
    array_name: str = "PROBED_LABEL_DISTANCES_MM",
) -> Tuple[List[float], vtk.vtkPolyData]:
    """
    Probe image values at the given point locations and attach
    the results in place as a scalar array to the input point set.
    """
    probed_vals = list()

    probed_vals_array = vtk.vtkFloatArray()
    probed_vals_array.SetName(array_name)
    probed_vals_array.SetNumberOfComponents(1)

    for point_id in range(input_mesh.GetNumberOfPoints()):
        point_loc = input_mesh.GetPoint(point_id)
        probed_val = probe_image_at_point(distance_image, point_loc)
        probed_vals_array.InsertNextTuple([probed_val])
        probed_vals.append(probed_val)

    input_mesh.GetPointData().AddArray(probed_vals_array)
    input_mesh.GetPointData().SetActiveScalars(array_name)
    return probed_vals, input_mesh


def read_write_probe_image(
    distance_image: itk.Image, mesh_filepath: str, dry_run: bool = False
) -> Tuple[List[float], vtk.vtkPolyData]:
    """
    Helper method to run the following routine:
    1. Read in a .vtp mesh
    2. Probe an image with the mesh vertices
    3. Attach probed data to mesh vertices
    4. Write out to the same .vtp file
    5. Write out point distances as a numpy array
    """
    import numpy as np

    distances_output_filename = (
        f"{os.path.splitext(mesh_filepath)[0]}_distances.csv"
    )

    print(f"Updating mesh with point distances (mm) at {mesh_filepath}")
    print(f"Distances will be written to {distances_output_filename}")
    if dry_run:
        return ([], vtk.vtkPolyData())

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(mesh_filepath)
    reader.Update()
    probed_vals, mesh = probe_image_vtk(distance_image, reader.GetOutput())
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(mesh_filepath)
    writer.SetInputData(mesh)
    writer.Update()

    np.savetxt(
        distances_output_filename,
        X=np.array(probed_vals),
        delimiter="\n",
        header="Distance to label boundary (mm)",
    )

    return probed_vals, mesh


def describe_list(point_distances):
    """Print a short statistical summary of list of one-dimensional values"""
    import pandas as pd

    df = pd.DataFrame(point_distances)
    print(df.describe())
