#!/usr/bin/env python3

# Purpose: Simple pytest to validate that module can be loaded.

import sys
sys.path.append("src/aind_ccf_alignment_experiments")

import itk
itk.auto_progress(2)

def test_loadmodule():
    import label_methods
    import point_distance
    import postprocess_cli
    import registration_methods
    import url
