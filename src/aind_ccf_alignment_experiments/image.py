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
    HALF_VOXEL_STEP = 0.5
    dimension = image.GetImageDimension()
    lower_index = np.array(
        image.GetLargestPossibleRegion().GetIndex(), dtype=np.float32
    )
    lower_index -= [-1 * HALF_VOXEL_STEP] * dimension
    upper_index = np.array(
        image.GetLargestPossibleRegion().GetUpperIndex(), dtype=np.float32
    )
    upper_index += [HALF_VOXEL_STEP] * dimension

    image_bounds = np.array(
        [
            image.TransformContinuousIndexToPhysicalPoint(
                arr_to_continuous_index(lower_index)
            ),
            image.TransformContinuousIndexToPhysicalPoint(
                arr_to_continuous_index(upper_index)
            ),
        ]
    )

    if transform:
        image_bounds = np.array(
            [transform.TransformPoint(pt) for pt in image_bounds]
        )

    return (np.min(image_bounds, axis=0), np.max(image_bounds, axis=0))


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


###############################################################################
# Image block streaming helpers
###############################################################################


def block_to_physical_size(
    block_size: npt.ArrayLike,
    ref_image: itk.Image,
    transform: itk.Transform = None,
) -> npt.ArrayLike:
    """Convert from voxel block size to corresponding size in physical space"""
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
    assert block_region.ndim == 2 and block_region.shape == (2, 3)

    index_to_physical_func = (
        lambda row: ref_image.TransformIndexToPhysicalPoint(
            [int(val) for val in row]
        )
    )
    if not transform:
        return np.apply_along_axis(index_to_physical_func, 1, block_region)

    tfm_func = lambda row: transform.TransformPoint(
        index_to_physical_func(row)
    )
    return np.apply_along_axis(tfm_func, 1, block_region)


def physical_to_block_region(
    physical_region: npt.ArrayLike, ref_image: itk.Image
) -> npt.ArrayLike:
    """Convert from physical region to corresponding voxel block"""
    # block region is a 2x3 matrix where row 0 is the lower bound and row 1 is the upper bound
    assert physical_region.ndim == 2 and physical_region.shape == (2, 3)

    physical_to_index_func = (
        lambda row: ref_image.TransformPhysicalPointToIndex(
            [float(val) for val in row]
        )
    )
    return np.apply_along_axis(physical_to_index_func, 1, physical_region)


def block_to_image_region(block_region: npt.ArrayLike) -> itk.ImageRegion[3]:
    """Convert from 2x3 bounds representation to itkImageRegion representation"""
    lower_index = [int(val) for val in np.min(block_region, axis=0)]
    upper_index = [int(val) for val in np.max(block_region, axis=0)]

    region = itk.ImageRegion[3]()
    region.SetIndex(lower_index)
    region.SetUpperIndex(upper_index)
    return region


def image_to_block_region(image_region: itk.ImageRegion[3]) -> npt.ArrayLike:
    """Convert from itkImageRegion to 2x3 bounds representation"""
    return np.array([image_region.GetIndex(), image_region.GetUpperIndex()])


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
    image_region: npt.ArrayLike, ref_image: itk.Image
) -> itk.ImageRegion[3]:
    """Convert from physical region to image region"""
    return block_to_physical_region(
        block_region=image_to_block_region(
            image_region=image_region, ref_image=ref_image
        )
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
    crop_to_target: bool = False,
) -> npt.ArrayLike:
    """
    Given a voxel region in source image space, compute the corresponding
    voxel region with physical alignment in target image space.
    """
    target_region = physical_to_block_region(
        physical_region=block_to_physical_region(
            block_region=block_region, ref_image=src_image
        ),
        ref_image=target_image,
    )

    if crop_to_target:
        image_region = block_to_image_region(block_region=target_region)
        image_region.Crop(target_image.GetLargestPossibleRegion())
        target_region = image_to_block_region(image_region=image_region)

    return target_region
