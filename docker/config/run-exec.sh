#!/bin/bash
# Default commands for executor container
# conda activate fedscale

# Setup fedscale package
conda run -n fedscale --no-capture-output pip install -e .

# Run executor
conda run -n fedscale --no-capture-output python3 fedscale/core/execution/executor.py --use_container