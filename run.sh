#!/bin/bash

# Exit on error
set -e

# Initialize conda
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Activate the conda environment
conda activate dzambala

# Resolve the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Change to the script directory
cd "$SCRIPT_DIR"

# Use the Python from the currently activated conda environment
PYTHON_CMD="python"
PYTHON_SCRIPT="./pytorch/test.py"

# Run the Python script with all arguments under group 'inet'
sg inet -c "$PYTHON_CMD $PYTHON_SCRIPT \"$@\""