#!/usr/bin/env python3

# Purpose: Validate registration utilities

import os
import sys

import numpy as np
import itk

sys.path.append(f"{os.getcwd()}/src")
import aind_ccf_alignment_experiments.registration_methods as aind_registration_methods


def test_flatten_composite_transform():
    t_inner = itk.CompositeTransform[itk.D, 3].New()
    translation1 = itk.TranslationTransform[itk.D, 3].New()
    translation1.Translate([1, 0, 0])
    t_inner.AppendTransform(translation1)
    translation2 = itk.TranslationTransform[itk.D, 3].New()
    translation2.Translate([2, 0, 0])
    t_inner.AppendTransform(translation2)
    t_outer = itk.CompositeTransform[itk.D, 3].New()
    translation3 = itk.TranslationTransform[itk.D, 3].New()
    translation3.Translate([3, 0, 0])
    t_outer.AppendTransform(translation3)
    t_outer.AppendTransform(t_inner)

    flattened_transform = (
        aind_registration_methods.flatten_composite_transform(t_outer)
    )
    assert flattened_transform.GetNumberOfTransforms() == 3
