#!/usr/bin/env python3

"""
Helpers for driving registration routines with ITK and ITKElastix
"""

from typing import List, Tuple, Dict

import itk
import numpy as np


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
    lower_index = itk.ContinuousIndex[itk.D, dimension]()
    lower_index.Fill(-1 * HALF_VOXEL_STEP)
    upper_index = itk.ContinuousIndex[itk.D, dimension]()
    for dim in range(dimension):
        upper_index.SetElement(dim, itk.size(image)[dim] + HALF_VOXEL_STEP)

    image_bounds = np.array(
        [
            image.TransformContinuousIndexToPhysicalPoint(lower_index),
            image.TransformContinuousIndexToPhysicalPoint(upper_index),
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


def compute_initial_translation(
    source_image: itk.Image, target_image: itk.Image
) -> Tuple[itk.TranslationTransform, itk.Image]:
    """
    Compute an initial translation to overlay the source image on the target image.
    """

    init_transform = itk.VersorRigid3DTransform[
        itk.D
    ].New()  # Represents 3D rigid transformation with unit quaternion
    init_transform.SetIdentity()

    transform_initializer = itk.CenteredVersorTransformInitializer[
        type(target_image), type(source_image)
    ].New()
    transform_initializer.SetFixedImage(target_image)
    transform_initializer.SetMovingImage(source_image)
    transform_initializer.SetTransform(init_transform)
    transform_initializer.GeometryOn()  # We compute translation between the center of each image
    transform_initializer.ComputeRotationOff()  # We have previously verified that spatial orientation aligns

    transform_initializer.InitializeTransform()

    # initial transform maps from the fixed image to the moving image
    # per ITK ResampleImageFilter convention
    translation_transform = itk.TranslationTransform[itk.D, 3].New()
    translation_transform.Translate(init_transform.TransformPoint([0, 0, 0]))

    # Apply translation without resampling the image by updating the image origin directly
    change_information_filter = itk.ChangeInformationImageFilter[
        type(source_image)
    ].New()
    change_information_filter.SetInput(source_image)
    change_information_filter.SetOutputOrigin(
        translation_transform.GetInverseTransform().TransformPoint(
            itk.origin(source_image)
        )
    )
    change_information_filter.ChangeOriginOn()
    change_information_filter.Update()

    return translation_transform, change_information_filter.GetOutput()


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
    verbose=False,
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

    if verbose:
        print(f"Register with parameter object:{parameter_object}")

    registration_method = itk.ElastixRegistrationMethod[
        type(target_image), type(source_image)
    ].New(
        fixed_image=target_image,
        moving_image=source_image,
        parameter_object=parameter_object,
        log_to_console=verbose,
    )

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
