#!/bin/bash

echo "In etc/setup/setup_ci_tests.sh"

# Check OS-type and resolve SCRIPT directory
echo -e "\033[34m[1/2]\033[0m Setting up environment with OS type: $OSTYPE"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Resolve script and target locations
testFolder="$(dirname "$(dirname "$(dirname "$SCRIPT")")")/tests"
downloadScript="$testFolder/etc/benchmark_scripts/download_ci_dataset.py"
controlDatasetDir="$testFolder/data/control-dataset"

# Control the dataset from Google Drive
if [ ! -d "$controlDatasetDir" ]; then
    echo -e "\033[34m[2/2]\033[0m Downloading the readily preprocessed control dataset"
    sudo -E env PATH="$PATH" PYTHONPATH="$PYTHONPATH" python "$downloadScript"
else
    echo -e "\033[34m[2/2]\033[0m Control dataset already exists"
fi