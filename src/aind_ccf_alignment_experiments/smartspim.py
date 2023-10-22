#!/usr/bin/env python3

"""
Methods for working with SmartSPIM lightsheet images provided by 
the Allen Institute for Neural Dynamics (AIND) on AWS S3.

Deals in particular with loading SmartSPIM data as an ITK image:
- Compose physically correct metadata (orientation, origing, spacing, size)
  representing SmartSPIM acquisition space
- Stream SmartSPIM subregions to fit on disk
"""

import logging
import json

import itk
import s3fs
import zarr.hierarchy
import numpy as np
import numpy.typing as npt

from .url import get_s3_https_bucket_data_url

logger = logging.getLogger(__name__)

# We have foreknowledge that 't' and 'c' axes are empty (size 1)
# ITK and OME-Zarr/numpy access conventions are reversed
ITK_SPATIAL_AXES = ("x", "y", "z")

# By convention ITK operations take place in
# right-to-left, anterior-to-posterior, inferior-to-superior (LPS) space.
ITK_TARGET_SPACE = (
    itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_LPS
)

# itk.OMEZarrNGFFImageIO (v0.1.7) does not yet perform axis unit conversions.
# By convention ITK operations typically assume millimeter spacing,
# whereas AIND SmartSPIM volumes typically have spacings encoded in micrometers.
# TODO spacing conversion may be unnecessary in later versions of "ITKIOOMEZarrNGFF".
# Tracked in https://github.com/InsightSoftwareConsortium/ITKIOOMEZarrNGFF/issues/42
SMARTSPIM_TO_ITK_SPACING = 1e-3


def make_smartspim_stream_reader(
    bucket_zarr_path: str,
    sample_level: int,
    fs: s3fs.S3FileSystem = s3fs.S3FileSystem(anon=True),
) -> itk.ImageFileReader[itk.Image[itk.F, 3]]:
    """
    Initialize an itk.ImageFileReader that can load requested image subregions from AWS.

    The reader image output will have fully populated metadata loaded from
    OME-Zarr headers and the bespoke SmartSPIM `acquisition.json` header file.
    The buffered (loaded) region of the image will initially have size 0
    and can be loaded with `reader.Update`.
    """
    smartspim_reader = get_smartspim_image_source(
        bucket_zarr_path, sample_level
    )

    imageio = itk.OMEZarrNGFFImageIO.cast(smartspim_reader.GetImageIO())
    imageio.SetTimeIndex(0)
    imageio.SetChannelIndex(0)
    smartspim_reader.SetImageIO(imageio)

    apply_smartspim_metadata(
        smartspim_reader.GetOutput(),
        sample_channel_bucket=bucket_zarr_path,
        dataset_id=sample_level,
        fs=fs,
    )
    smartspim_reader.GetOutput().SetRequestedRegion(itk.ImageRegion[3]([0,0,0]))
    return smartspim_reader


def get_smartspim_image_source(
    sample_channel_bucket: str, dataset_id: int
) -> itk.ImageFileReader[itk.Image[itk.F, 3]]:
    """
    Parse acquisition metadata directly from OME-Zarr headers.

    OME-Zarr v0.4 does not encode acquisition direction. The output volume
    may not be properly oriented with respect to anatomical axes.
    """
    imageio = itk.OMEZarrNGFFImageIO.New()
    imageio.SetDatasetIndex(dataset_id)
    # Assume SmartSPIM data is in t,c,z,y,x format and we want only the first t/c slices
    imageio.SetTimeIndex(0)
    imageio.SetChannelIndex(0)

    image_reader = itk.ImageFileReader[itk.Image[itk.F, 3]].New()
    image_reader.SetImageIO(imageio)
    image_reader.SetFileName(
        get_s3_https_bucket_data_url(sample_channel_bucket)
    )
    image_reader.UpdateOutputInformation()

    # Account for micron-to-millimeter spacing adjustments.
    # TODO this may be unnecessary with later versions (>0.1.7) of ITKIOOMEZarrNGFF
    # Tracked in https://github.com/InsightSoftwareConsortium/ITKIOOMEZarrNGFF/issues/42
    spacing = np.array(itk.spacing(image_reader.GetOutput()))
    spacing *= SMARTSPIM_TO_ITK_SPACING
    image_reader.GetOutput().SetSpacing(spacing)

    return image_reader


def apply_smartspim_metadata(
    image: itk.Image,
    sample_channel_bucket: str,
    dataset_id: int,
    fs: s3fs.S3FileSystem,
) -> itk.Image:
    """Apply SmartSPIM metadata to a SmartSPIM image in place"""
    # OME-Zarr SmartSPIM storage reflects origin and spacing metadata
    reader = get_smartspim_image_source(
        sample_channel_bucket=sample_channel_bucket, dataset_id=dataset_id
    )
    unbuffered_smartspim_image = reader.GetOutput()

    # acquisition.json SmartSPIM header reflects direction
    acquisition_metadata = get_smartspim_acquisition_metadata(
        sample_channel_bucket=sample_channel_bucket, fs=fs
    )
    itk_acquisition_space = parse_acquisition_space(acquisition_metadata)

    # Update metadata to reflect the target space.
    # Origin, spacing, and direction may be updated to reflect orientation.
    orient_filter = itk.OrientImageFilter.New(unbuffered_smartspim_image)
    orient_filter.SetGivenCoordinateOrientation(itk_acquisition_space)
    orient_filter.SetDesiredCoordinateOrientation(ITK_TARGET_SPACE)
    orient_filter.UpdateOutputInformation()
    image.CopyInformation(orient_filter.GetOutput())
    return image


def get_smartspim_acquisition_metadata(
    sample_channel_bucket: str, fs: s3fs.S3FileSystem
) -> dict:
    """
    Fetch SmartSPIM JSON acquisition metadata from AIND S3 storage.
    """
    ACQUISITION_SCHEMA_VERSION = "0.4.3"
    acquisition_filepath = (
        sample_channel_bucket.split("_stitched")[0] + "/acquisition.json"
    )
    with fs.open(acquisition_filepath, "r") as f:
        acquisition_metadata = json.loads(f.read())
    assert acquisition_metadata["schema_version"] == ACQUISITION_SCHEMA_VERSION
    return acquisition_metadata


def parse_acquisition_space(acquisition_metadata: dict, verbose: bool = False):
    # Acquisition spatial axis directions are encoded in ITK as a bitmap composition
    # of enumerated values in tertiary, secondary, primary order.
    # https://github.com/InsightSoftwareConsortium/ITK/blob/cce48b3cfa5c8f41be0bf7393183132ab34f1a06/Modules/Core/Common/include/itkSpatialOrientation.h#L111-L112

    ITK_COORDINATE_TERMS = {
        "anterior": itk.SpatialOrientationEnums.CoordinateTerms_ITK_COORDINATE_Anterior,
        "inferior": itk.SpatialOrientationEnums.CoordinateTerms_ITK_COORDINATE_Inferior,
        "left": itk.SpatialOrientationEnums.CoordinateTerms_ITK_COORDINATE_Left,
        "posterior": itk.SpatialOrientationEnums.CoordinateTerms_ITK_COORDINATE_Posterior,
        "right": itk.SpatialOrientationEnums.CoordinateTerms_ITK_COORDINATE_Right,
        "superior": itk.SpatialOrientationEnums.CoordinateTerms_ITK_COORDINATE_Superior,
    }

    acquisition_space_name = []
    itk_direction_enum = 0

    for axis_name in reversed(ITK_SPATIAL_AXES):
        axis = next(
            el
            for el in acquisition_metadata["axes"]
            if el["name"].lower() == axis_name
        )
        axis_direction = axis["direction"].split("_to_")[1]
        acquisition_space_name.append(axis_direction)

        itk_direction_enum <<= 8
        itk_direction_enum |= ITK_COORDINATE_TERMS[axis_direction]

    if verbose:
        print(
            f'{"-".join(reversed(acquisition_space_name))}: {itk_direction_enum}'
        )

        # Values can be manually double checked against ITK coordinate space enums, i.e.
        print(
            f"ITK RAI: {itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAI}"
        )

    return itk_direction_enum
