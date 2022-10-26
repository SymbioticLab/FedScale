#!/bin/bash
# Default commands for aggregator container
# conda activate fedscale

# Run aggregator
conda run -n fedscale --no-capture-output python3 examples/containerization/aggregator_ctnr.py