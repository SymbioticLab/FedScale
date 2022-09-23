#!/bin/bash
# Default commands for executor container
# conda activate fedscale

# Run executor
conda run -n fedscale --no-capture-output python3 examples/containerization/executor_ctnr.py