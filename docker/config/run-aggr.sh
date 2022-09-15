#!/bin/bash
# Default commands for aggregator container
# conda activate fedscale

# Setup fedscale package
conda run -n fedscale --no-capture-output pip install -e .

# Run aggregator
conda run -n fedscale --no-capture-output python3 fedscale/core/aggregation/aggregator.py --use_container