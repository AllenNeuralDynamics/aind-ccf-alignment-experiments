#!/usr/bin/env python3

# Purpose: Simple pytest to validate that module can be loaded.

import sys

sys.path.append("src")

import itk

itk.auto_progress(2)


def test_loadmodule():
    import aind_ccf_alignment_experiments.label_methods
    import aind_ccf_alignment_experiments.point_distance
    import aind_ccf_alignment_experiments.postprocess_cli
    import aind_ccf_alignment_experiments.registration_methods
    import aind_ccf_alignment_experiments.url
