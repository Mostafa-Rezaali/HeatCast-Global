#!/usr/bin/env python3
"""
Probe exported teacher-forced latent rollout maps with the same 1x1 gate.

Run:
    python3 -u probe_latent_rollout_maps.py --device cuda --batch_size 8
"""

import sys

from met_jepa import LATENT_ROLLOUT_MAPS_PATH, LATENT_ROLLOUT_META_PATH, LATENT_ROLLOUT_PROBE_REPORT


def add_default_arg(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


add_default_arg("--maps_path", LATENT_ROLLOUT_MAPS_PATH)
add_default_arg("--meta_path", LATENT_ROLLOUT_META_PATH)
add_default_arg("--report", LATENT_ROLLOUT_PROBE_REPORT)
add_default_arg("--title", "Teacher-Forced Latent Rollout Probe Gate")

from probe_jepa_maps import main  # noqa: E402


if __name__ == "__main__":
    main()
