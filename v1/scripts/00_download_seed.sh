#!/bin/bash
# TeleopQ v1 - step 0: download NVIDIA's annotated GR-1 seed demos (5 demos, ~0.9 MB).
# Run from the Isaac Lab root (/workspace/isaaclab on the Brev Isaac Launchable).
set -euo pipefail
mkdir -p datasets
wget -O datasets/dataset_annotated_gr1.hdf5 \
  https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Mimic/pick_place_datasets/dataset_annotated_gr1.hdf5
ls -lh datasets/dataset_annotated_gr1.hdf5
