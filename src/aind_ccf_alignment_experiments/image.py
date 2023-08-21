#!/usr/bin/env python3

"""General ITK image utilities for mapping between voxel and physical spaces."""

import itk
import numpy as np
import numpy.typing as npt


def arr_to_continuous_index(arr: npt.ArrayLike) -> itk.ContinuousIndex:
    assert arr.ndim == 1
    itk_index = itk.ContinuousIndex[itk.D, arr.shape[0]]()
    for index in range(arr.shape[0]):
        itk_index.SetElement(index, float(arr[index]))
    return itk_index


def get_sample_bounds(image: itk.Image, transform: itk.Transform = None):
    """Get the physical boundaries of the space sampled by the ITK image.
    Each voxel in an ITK image is considered to be a sample of the spatial
    volume occupied by that voxel taken at the spatial center of the volume.
    The physical point returned at each discrete voxel coordinate is
    considered to be the physical location of the sample point. We adjust by
    half a voxel in each direction to get the bounds of the space sampled
    by the image.
    """
    return image_to_physical_region(
        image.GetLargestPossibleRegion(),
        ref_image=image,
        src_transform=transform,
    )


def get_physical_size(image: itk.Image, transform: itk.Transform = None):
    """Get the distance along each size of the physical space sampled by the image"""
    bounds = get_sample_bounds(image, transform)
    return np.absolute(bounds[1] - bounds[0])


def get_physical_midpoint(
    image: itk.Image, transform: itk.Transform = None
) -> npt.ArrayLike:
    """Get the physical midpoint of the image"""
    bounds = get_sample_bounds(image, transform)
    return np.mean(bounds, axis=0)


def block_to_itk_image(
    data: npt.ArrayLike, start_index: npt.ArrayLike, reference_image: itk.Image
) -> itk.Image:
    """Return an ITK image view into an array block.
    `data` -> the mapped block data
    `start_index` -> the image index of the first voxel in the data array in ITK access order.
    `reference_image` -> reference ITK image metadata for the image,
        may have empty buffered region.
    """
    block_image = itk.image_view_from_array(data)

    buffer_offset = [int(val) for val in start_index]
    block_image = itk.change_information_image_filter(
        block_image,
        change_region=True,
        output_offset=buffer_offset,
        change_origin=True,
        output_origin=reference_image.GetOrigin(),
        change_spacing=True,
        output_spacing=reference_image.GetSpacing(),
        change_direction=True,
        output_direction=reference_image.GetDirection(),
    )
    return itk.extract_image_filter(
        block_image, extraction_region=block_image.GetBufferedRegion()
    )


###############################################################################
# Image block streaming helpers
###############################################################################

# Terms:
# - "block": A representation in voxel space with integer image access.
# - "physical": A representation in physical space with 3D floating-point representation.
#
# - "block region": a 2x3 voxel array representing [lower,upper) voxel bounds
#                   in ITK access order.
#                   To mimic NumPy indexing the lower bound is inclusive and the upper
#                   bound is one greater than the last included voxel index.
#                   If "k" is fastest and "i" is slowest:
#                   [ [ lower_k, lower_j, lower_i ]
#                       upper_k, upper_j, upper_i ] ]
#
# - "physical region": a 2x3 voxel array representing inclusive upper and lower bounds in
#                      physical space.
#                      [ [ lower_x, lower_y, lower_z ]
#                          upper_x, upper_y, upper_z ] ]
#
# - "ITK region": an `itk.ImageRegion[3]` representation of a block region.
#                 itk.ImageRegion[3]( [ [lower_k, lower_j, lower_i], [size_k, size_j, size_i] ])


def block_to_physical_size(
    block_size: npt.ArrayLike,
    ref_image: itk.Image,
    transform: itk.Transform = None,
) -> npt.ArrayLike:
    """
    Convert from voxel block size to corresponding size in physical space.

    Naive transform approach assumes that both the input and output regions
    are constrained along x/y/z planes aligned at two point extremes.
    May not be suitable for deformable regions.
    """
    block_index = [int(x) for x in block_size]

    if transform:
        return np.abs(
            transform.TransformPoint(
                ref_image.TransformIndexToPhysicalPoint(block_index)
            )
            - transform.TransformPoint(itk.origin(ref_image))
        )

    return np.abs(
        ref_image.TransformIndexToPhysicalPoint(block_index)
        - itk.origin(ref_image)
    )


def physical_to_block_size(
    physical_size: npt.ArrayLike, ref_image: itk.Image
) -> npt.ArrayLike:
    """Convert from physical size to corresponding voxel size"""
    return np.abs(
        ref_image.TransformPhysicalPointToIndex(
            itk.origin(ref_image) + physical_size
        )
    )


def block_to_physical_region(
    block_region: npt.ArrayLike,
    ref_image: itk.Image,
    transform: itk.Transform = None,
) -> npt.ArrayLike:
    """Convert from voxel region to corresponding physical space"""
    # block region is a 2x3 matrix where row 0 is the lower bound and row 1 is the upper bound
    HALF_VOXEL_STEP = 0.5

    assert block_region.ndim == 2 and block_region.shape == (2, 3)
    block_region = np.array(
        [np.min(block_region, axis=0), np.max(block_region, axis=0)]
    )

    adjusted_block_region = block_region - HALF_VOXEL_STEP

    index_to_physical_func = (
        lambda row: ref_image.TransformContinuousIndexToPhysicalPoint(
            arr_to_continuous_index(row)
        )
    )
    physical_region = np.apply_along_axis(
        index_to_physical_func, 1, adjusted_block_region
    )

    if transform:
        tfm_func = lambda row: transform.TransformPoint(row)
        physical_region = np.apply_along_axis(tfm_func, 1, physical_region)

    return np.array(
        [np.min(physical_region, axis=0), np.max(physical_region, axis=0)]
    )


def physical_to_block_region(
    physical_region: npt.ArrayLike, ref_image: itk.Image
) -> npt.ArrayLike:
    """
    Convert from physical region to corresponding voxel block
    """
    HALF_VOXEL_STEP = 0.5

    # block region is a 2x3 matrix where row 0 is the lower bound and row 1 is the upper bound
    assert physical_region.ndim == 2 and physical_region.shape == (2, 3)

    physical_to_index_func = (
        lambda row: ref_image.TransformPhysicalPointToContinuousIndex(
            [float(val) for val in row]
        )
    )
    block_region = np.apply_along_axis(
        physical_to_index_func, 1, physical_region
    )
    adjusted_block_region = np.array(
        [np.min(block_region, axis=0), np.max(block_region, axis=0)]
    )
    return adjusted_block_region + HALF_VOXEL_STEP


def block_to_image_region(block_region: npt.ArrayLike) -> itk.ImageRegion[3]:
    """Convert from 2x3 bounds representation to itkImageRegion representation"""
    lower_index = [int(val) for val in np.min(block_region, axis=0)]
    upper_index = [int(val) - 1 for val in np.max(block_region, axis=0)]

    region = itk.ImageRegion[3]()
    region.SetIndex(lower_index)
    region.SetUpperIndex(upper_index)
    return region


def image_to_block_region(image_region: itk.ImageRegion[3]) -> npt.ArrayLike:
    """Convert from itkImageRegion to 2x3 bounds representation"""
    return np.array(
        [image_region.GetIndex(), np.array(image_region.GetUpperIndex()) + 1]
    )


def physical_to_image_region(
    physical_region: npt.ArrayLike, ref_image: itk.Image
) -> itk.ImageRegion[3]:
    """Convert from physical region to image region"""
    return block_to_image_region(
        block_region=physical_to_block_region(
            physical_region=physical_region, ref_image=ref_image
        )
    )


def image_to_physical_region(
    image_region: npt.ArrayLike,
    ref_image: itk.Image,
    src_transform: itk.Transform = None,
) -> itk.ImageRegion[3]:
    """Convert from physical region to image region"""
    return block_to_physical_region(
        block_region=image_to_block_region(image_region=image_region),
        ref_image=ref_image,
        transform=src_transform,
    )


def get_target_block_size(
    block_size: npt.ArrayLike, src_image: itk.Image, target_image: itk.Image
) -> npt.ArrayLike:
    """
    Given a voxel region size in source image space, compute the corresponding
    voxel region size with physical alignment in target image space.
    """
    return physical_to_block_size(
        block_to_physical_size(block_size, src_image), target_image
    )


def get_target_block_region(
    block_region: npt.ArrayLike,
    src_image: itk.Image,
    target_image: itk.Image,
    src_transform: itk.Transform = None,
    crop_to_target: bool = False,
) -> npt.ArrayLike:
    """
    Given a voxel region in source image space, compute the corresponding
    voxel region with physical alignment in target image space.
    """
    target_region = physical_to_block_region(
        physical_region=block_to_physical_region(
            block_region=block_region,
            ref_image=src_image,
            transform=src_transform,
        ),
        ref_image=target_image,
    )

    if crop_to_target:
        # TODO can we preserve continuous index input?
        image_region = block_to_image_region(block_region=target_region)
        image_region.Crop(target_image.GetLargestPossibleRegion())
        target_region = image_to_block_region(image_region=image_region)

    return target_region
