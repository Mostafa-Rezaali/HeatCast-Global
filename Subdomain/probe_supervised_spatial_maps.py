#!/usr/bin/env python3
"""
Probe exported supervised spatial maps with the same cheap 1x1 decoder gate.

Run:
    python3 -u probe_supervised_spatial_maps.py --device cuda --batch_size 8
"""

import sys

from met_jepa import SUPERVISED_MAPS_PATH, SUPERVISED_META_PATH, SUPERVISED_PROBE_REPORT


def add_default_arg(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


add_default_arg("--maps_path", SUPERVISED_MAPS_PATH)
add_default_arg("--meta_path", SUPERVISED_META_PATH)
add_default_arg("--report", SUPERVISED_PROBE_REPORT)
add_default_arg("--title", "Supervised Spatial Map Probe Gate")

from probe_jepa_maps import main  # noqa: E402


if __name__ == "__main__":
    main()
