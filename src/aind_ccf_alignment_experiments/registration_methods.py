#!/usr/bin/env python3

"""
Helpers for driving registration routines with ITK and ITKElastix
"""
import os
from typing import List, Tuple, Dict

import itk
import numpy as np

from .image import get_sample_bounds, get_physical_size, get_physical_midpoint


def compute_initial_translation(
    source_image: itk.Image, target_image: itk.Image
) -> itk.TranslationTransform[itk.D, 3]:
    """
    Compute the initial overlap transform as the translation to sample
    from the center of the target image to the center of the source image.

    Assumes content is centered in source and target images.
    """
    target_midpoint = get_physical_midpoint(target_image)
    source_midpoint = get_physical_midpoint(source_image)

    translation_transform = itk.TranslationTransform[itk.D, 3].New()
    translation_transform.Translate(source_midpoint - target_midpoint)

    return translation_transform


def get_elx_itk_transforms(
    registration_method: itk.ElastixRegistrationMethod,
    itk_transform_types: List[itk.Transform],
) -> itk.CompositeTransform:
    """
    Convert Elastix registration results to an ITK composite transform stack
    of known, corresponding types.
    """
    value_type = itk.D
    dimension = 3

    if registration_method.GetNumberOfTransforms() != len(itk_transform_types):
        raise ValueError(
            f"Expected {registration_method.GetNumberOfTransforms()} mappings"
            f"but {len(itk_transform_types)} were found"
        )

    itk_composite_transform = itk.CompositeTransform[
        value_type, dimension
    ].New()

    try:
        for transform_index, itk_transform_type in enumerate(
            itk_transform_types
        ):
            elx_transform = registration_method.GetNthTransform(
                transform_index
            )
            itk_base_transform = registration_method.ConvertToItkTransform(
                elx_transform
            )
            itk_transform = itk_transform_type.cast(itk_base_transform)
            itk_composite_transform.AddTransform(itk_transform)
    except RuntimeError as e:  # handle bad cast
        print(e)
        return None

    return itk_composite_transform


def get_elx_parameter_maps(
    registration_method: itk.ElastixRegistrationMethod,
) -> List[itk.ParameterObject]:
    """
    Return a series of transform parameter results from Elastix registration
    """
    transform_parameter_object = (
        registration_method.GetTransformParameterObject()
    )
    output_parameter_maps = [
        transform_parameter_object.GetParameterMap(parameter_map_index)
        for parameter_map_index in range(
            transform_parameter_object.GetNumberOfParameterMaps()
        )
    ]
    return output_parameter_maps


def make_default_elx_parameter_object() -> itk.ParameterObject:
    """
    Generate a default set of parameters for Elastix registration
    """
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(
        parameter_object.GetDefaultParameterMap("rigid")
    )
    parameter_object.AddParameterMap(
        parameter_object.GetDefaultParameterMap("affine")
    )

    bspline_map = parameter_object.GetDefaultParameterMap("bspline")
    bspline_map["FinalGridSpacingInPhysicalUnits"] = ("0.5000",)
    parameter_object.AddParameterMap(bspline_map)
    return parameter_object


def register_elastix(
    source_image: itk.Image,
    target_image: itk.Image,
    parameter_object: itk.ParameterObject = None,
    itk_transform_types: List[type] = None,
    log_filepath: str = None,
    verbose: bool = False,
) -> Tuple[itk.CompositeTransform, itk.Image, itk.ElastixRegistrationMethod]:
    """
    Compute a series of ITKElastix transforms mapping
    from the source image to the target image.
    """
    BSPLINE_ORDER = 3
    DIMENSION = 3

    if not "ElastixRegistrationMethod" in dir(itk):
        raise KeyError(
            "Elastix methods not found, please pip install itk-elastix"
        )

    if not parameter_object:
        parameter_object = make_default_elx_parameter_object()
    if not itk_transform_types:
        itk_transform_types = [
            itk.BSplineTransform[itk.D, DIMENSION, BSPLINE_ORDER],
            itk.AffineTransform[itk.D, DIMENSION],
            itk.Euler3DTransform[itk.D],
        ]
    if log_filepath:
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    if verbose:
        print(f"Register with parameter object:{parameter_object}")

    registration_method = itk.ElastixRegistrationMethod[
        type(target_image), type(source_image)
    ].New(
        fixed_image=target_image,
        moving_image=source_image,
        parameter_object=parameter_object,
    )

    if log_filepath:
        registration_method.SetLogToFile(True)
        registration_method.SetOutputDirectory(os.path.dirname(log_filepath))
        registration_method.SetLogFileName(os.path.basename(log_filepath))

    # Run registration with `itk-elastix`, may take a few minutes
    registration_method.Update()

    itk_composite_transform = get_elx_itk_transforms(
        registration_method, itk_transform_types
    )

    return (
        itk_composite_transform,
        registration_method.GetOutput(),
        registration_method,
    )


def resample_label_image(
    source_image: itk.Image,
    reference_image: itk.Image,
    transform: itk.Transform,
) -> itk.Image:
    """
    Resample a source label image into reference imace space with
    LabelImageGenericInterpolateImageFunction to avoid introducing
    artifacts from interpolation of discrete label values.
    """
    if not "LabelImageGenericInterpolateImageFunction" in dir(itk):
        raise KeyError(
            "Label interpolator not found. Please pip install itk-genericlabelinterpolator"
        )

    return itk.resample_image_filter(
        source_image,
        transform=transform,
        use_reference_image=True,
        reference_image=reference_image,
        interpolator=itk.LabelImageGenericInterpolateImageFunction[
            type(source_image)
        ].New(),
    )


def register_ants(
    source_image: itk.Image,
    target_image: itk.Image,
    tmp_dir: str,
    verbose: bool = True,
) -> Tuple[itk.CompositeTransform, Dict]:
    """
    Compute a series of transforms mapping from
    the source to target image domain using ANTSxPy
    """
    import ants
    from scipy.io import loadmat

    INPUT_TARGET_TMP_FILEPATH = f"{tmp_dir}/source_image_tmp.nii.gz"
    INPUT_SOURCE_TMP_FILEPATH = f"{tmp_dir}/target_image_tmp.nii.gz"

    itk.imwrite(source_image, INPUT_SOURCE_TMP_FILEPATH, compression=True)
    itk.imwrite(target_image, INPUT_TARGET_TMP_FILEPATH, compression=True)

    ants_source_image = ants.image_read(INPUT_SOURCE_TMP_FILEPATH)
    ants_target_image = ants.image_read(INPUT_TARGET_TMP_FILEPATH)

    # https://antspy.readthedocs.io/en/latest/registration.html
    # Initial parameters and script provided by the Allen Institute for Neural Dynamics

    ANTS_TRANSFORM_TYPE = "SyN"  # symmetric normalization
    ANTS_REG_ITERATIONS = [100, 10, 0]  # multiresolution parameters

    ants_result = ants.registration(
        fixed=ants_target_image,
        moving=ants_source_image,
        type_of_transform=ANTS_TRANSFORM_TYPE,
        reg_iterations=ANTS_REG_ITERATIONS,
        verbose=verbose,
    )

    if verbose:
        print(ants_result)

    # Load transforms from output disk location into ITK format
    assert (
        ants_result["fwdtransforms"][1] == ants_result["invtransforms"][0]
    )  # ants references same affine transform fwd/inv

    generic_affine_mat = loadmat(ants_result["fwdtransforms"][1])
    affine_transform = itk.AffineTransform[itk.D, 3].New()

    fixed_param = affine_transform.GetFixedParameters()
    for index, el in enumerate(generic_affine_mat["fixed"]):
        fixed_param.SetElement(index, float(el))
    affine_transform.SetFixedParameters(fixed_param)

    param = affine_transform.GetParameters()
    for index, el in enumerate(
        generic_affine_mat["AffineTransform_float_3_3"]
    ):
        param.SetElement(index, float(el))
    affine_transform.SetParameters(param)

    def cast_vector_image_to_double(vector_image: itk.Image) -> itk.Image:
        """
        64-bit floating point type ("double") required to compose transforms in ITK Python
        """
        arr = itk.array_view_from_image(vector_image)
        vector_image_d = itk.image_view_from_array(
            arr.astype(np.float64), is_vector=True
        )
        vector_image_d.CopyInformation(vector_image)
        return vector_image_d

    deformation_field_filepath = ants_result["fwdtransforms"][0]

    assert "Warp" in deformation_field_filepath
    assert "Inverse" not in deformation_field_filepath

    displacement_field_f = itk.imread(deformation_field_filepath)
    displacement_field = cast_vector_image_to_double(displacement_field_f)
    displacement_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
    displacement_transform.SetDisplacementField(displacement_field)

    composite_transform = itk.CompositeTransform[itk.D, 3].New()
    composite_transform.AddTransform(affine_transform)
    composite_transform.AddTransform(displacement_transform)

    return composite_transform, ants_result
