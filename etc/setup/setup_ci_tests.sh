#!/bin/bash

is_empty() {
    # This way because of possible gitignore
    local resourcesDir=$1
    if [ -d "$resourcesDir" ] && [ -z "$(find "$resourcesDir" -mindepth 1 -type f | grep -v '/\.gitignore$')" ]; then
        return 0
    else
        return 1
    fi
}

echo "In etc/setup/setup_ci_tests.sh"

# Check OS-type and resolve SCRIPT directory
echo -e "\033[34m[1/2]\033[0m Setting up environment with OS type: $OSTYPE"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# MIMIC-Demo dataset source and test directory locations
testFolder="$(dirname "$(dirname "$(dirname "$SCRIPT")")")/tests"

# Resolve script and target locations
destinationDir="$testFolder/data/"
demoDataDir="$destinationDir/mimiciii-demo/"
downloadScriptDemo="$testFolder/etc/benchmark_scripts/download_ci_demo_data.py"

# Get the reduced form demo dataset from Google Drive
if is_empty "$demoDataDir"; then
    echo -e "\033[34m[6/huregeil]\033[0m Downloading the reduced form demo dataset"
    sudo -E env PATH="$PATH" PYTHONPATH="$PYTHONPATH" python "$downloadScriptDemo"
else
    echo -e "\033[34m[6/huregeil]\033[0m Demo dataset already exists"
fi


# Resolve script and target locations
downloadScriptControl="$testFolder/etc/benchmark_scripts/download_ci_dataset.py"
controlDatasetDir="$testFolder/data/control-dataset"

# Get the control dataset from Google Drive
if is_empty "$controlDatasetDir"; then
    echo -e "\033[34m[6/huregeil]\033[0m Downloading the readily preprocessed control dataset"
    sudo -E env PATH="$PATH" PYTHONPATH="$PYTHONPATH" python "$downloadScriptControl"
else
    echo -e "\033[34m[6/huregeil]\033[0m Control dataset already exists"
fi
