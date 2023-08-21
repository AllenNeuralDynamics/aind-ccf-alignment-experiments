#!/usr/bin/env python3

# Purpose: Validate logic for transforming indices
# between image voxel space and physical space

import os
import sys

import numpy as np
import itk

sys.path.append(f"{os.getcwd()}/src")
import aind_ccf_alignment_experiments.image as aind_image


def make_test_image(size: np.array) -> itk.Image[itk.F, 3]:
    assert size.ndim == 1 and size.shape[0] == 3
    arr = np.zeros(np.flip(size), dtype=np.float32)
    image = itk.image_from_array(arr)
    assert all(
        [image_dim == dim for image_dim, dim in zip(itk.size(image), size)]
    )
    return image


def test_block_and_image_region():
    IMAGE_SHAPE = np.array([1, 2, 3])
    image = make_test_image(IMAGE_SHAPE)

    EXPECTED_LARGEST_REGION = np.array([[0, 0, 0], IMAGE_SHAPE])
    block_region = aind_image.image_to_block_region(
        image.GetLargestPossibleRegion()
    )
    assert np.all(
        block_region == EXPECTED_LARGEST_REGION
    ), f"Incorrect block region {block_region}"

    image_region = aind_image.block_to_image_region(block_region)
    assert (
        image_region == image.GetLargestPossibleRegion()
    ), f"Incorrect image region {image_region}"


def test_image_and_physical_region():
    IMAGE_SHAPE = np.array([1, 2, 3])
    image = make_test_image(IMAGE_SHAPE)
    image.SetSpacing([0.1, 1.0, 10.0])
    image.SetOrigin([-1, 0, 1])

    EXPECTED_PHYSICAL_REGION = np.array([[-1.05, -0.5, -4], [-0.95, 1.5, 26]])
    physical_region = aind_image.image_to_physical_region(
        image.GetLargestPossibleRegion(), image
    )
    assert np.all(
        physical_region == EXPECTED_PHYSICAL_REGION
    ), f"Incorrect physical region {physical_region}"

    image_region = aind_image.physical_to_image_region(physical_region, image)
    assert image_region == image.GetLargestPossibleRegion()

    transform = itk.TranslationTransform[itk.D, 3].New()
    transform.Translate([1.0] * 3)
    EXPECTED_PHYSICAL_REGION += 1.0
    physical_region = aind_image.image_to_physical_region(
        image.GetLargestPossibleRegion(), image, transform
    )
    assert np.all(physical_region == EXPECTED_PHYSICAL_REGION)


def test_transform_block_to_target():
    IMAGE_SHAPE = np.array([5, 5, 5])
    source_image = make_test_image(IMAGE_SHAPE)
    target_image = make_test_image(IMAGE_SHAPE)
    target_image.SetOrigin([-1, -1, -1])

    SOURCE_REGION = np.array([[0, 0, 0], [1, 1, 1]])
    EXPECTED_TARGET_REGION = np.array([[1, 1, 1], [2, 2, 2]])
    region = aind_image.get_target_block_region(
        block_region=SOURCE_REGION,
        src_image=source_image,
        target_image=target_image,
    )
    assert np.all(
        region == EXPECTED_TARGET_REGION
    ), f"Incorrect target region {region}"

    region = aind_image.get_target_block_region(
        block_region=region, src_image=target_image, target_image=source_image
    )
    assert np.all(region == SOURCE_REGION), f"Incorrect target region {region}"

    # Verify for initial transform mapping from source space to common space
    source_translation = itk.TranslationTransform[itk.D, 3].New()
    source_translation.Translate([1, 1, 1])
    EXPECTED_TARGET_REGION = np.array([[2, 2, 2], [3, 3, 3]])
    region = aind_image.get_target_block_region(
        block_region=SOURCE_REGION,
        src_image=source_image,
        target_image=target_image,
        src_transform=source_translation,
    )
    assert np.all(
        region == EXPECTED_TARGET_REGION
    ), f"Incorrect target region {region}"

    # Verify for crop
    SOURCE_REGION = np.array([[3, 3, 3], [5, 5, 5]])
    EXPECTED_TARGET_REGION = np.array([[4, 4, 4], [5, 5, 5]])
    region = aind_image.get_target_block_region(
        block_region=SOURCE_REGION,
        src_image=source_image,
        target_image=target_image,
        crop_to_target=True,
    )
    assert np.all(
        region == EXPECTED_TARGET_REGION
    ), f"Incorrect target region {region}"


def test_block_to_itk_image():
    IMAGE_SHAPE = np.array([5, 5, 5])
    source_image = make_test_image(IMAGE_SHAPE)
    source_image[1:3, 1:3, 1:3] = 1
    source_image.SetOrigin([-1, -1, -1])
    source_image.SetSpacing([2, 2, 2])

    LOWER_INDEX = np.array([1, 1, 1])
    UPPER_INDEX = np.array([4, 4, 4])
    output_image = aind_image.block_to_itk_image(
        data=source_image[
            LOWER_INDEX[0] : UPPER_INDEX[0],
            LOWER_INDEX[1] : UPPER_INDEX[1],
            LOWER_INDEX[2] : UPPER_INDEX[2],
        ],
        start_index=LOWER_INDEX,
        reference_image=source_image,
    )

    assert np.all(
        np.array(itk.size(output_image)) == UPPER_INDEX - LOWER_INDEX
    )
    buffered_region = output_image.GetBufferedRegion()
    assert np.all(np.array(buffered_region.GetIndex()) == LOWER_INDEX)
    assert np.all(np.array(buffered_region.GetUpperIndex()) == UPPER_INDEX - 1)
    assert np.all(
        np.array(output_image.GetDirection())
        == np.array(source_image.GetDirection())
    )
    assert itk.origin(output_image) == itk.origin(source_image)
    assert itk.spacing(output_image) == itk.spacing(source_image)

    EXPECTED_BOUNDS = np.array([[0, 0, 0], [6, 6, 6]])
    block_bounds = aind_image.get_sample_bounds(output_image)
    assert np.all(block_bounds == EXPECTED_BOUNDS)
