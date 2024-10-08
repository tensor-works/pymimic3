#!/bin/bash

is_empty() {
    # This way because of possible gitignore
    local resourcesDir=$1
    if [ -d "$resourcesDir" ] && [ -z "$(find "$resourcesDir" -mindepth 1 -type f | grep -v '/\.gitignore$')" ]; then
        echo "The directory contains only .gitignore or is empty."
    else
        echo "The directory contains files other than .gitignore."
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
sourceUrl="https://physionet.org/files/mimiciii-demo/1.4/"
testFolder="$(dirname "$(dirname "$(dirname "$SCRIPT")")")/tests"

# Download the MIMIC-III demo dataset from the web
destinationDir="$testFolder/data/"
convertScript="$testFolder/etc/benchmark_scripts/convert_columns.py"
demoDataDir="$destinationDir/mimiciii-demo/"

if is_empty "$demoDataDir"; then
    echo -e "\033[34m[3/huregeil]\033[0m Downloading the MIMIC-III demo dataset directory"
    sudo wget -r -N -c -np $sourceUrl -P $destinationDir
    
    # Correcting defaults of the demo dataset
    echo -e "\033[34m[4/huregeil]\033[0m Correcting headers of the MIMIC-III demo dataset"
    sudo mkdir $demoDataDir
    origCsvDir="$destinationDir/physionet.org/files/mimiciii-demo/1.4/"
    sudo cp $origCsvDir/* $demoDataDir
    sudo rm -rf $origCsvDir
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $convertScript $demoDataDir
else
    echo -e "\033[34m[3/huregeil]\033[0m MIMIC-III demo dataset directory already exists"
    echo -e "\033[34m[4/huregeil]\033[0m Header already corrected"
fi

# Download the MIMIC-III demo config files from original repo
resourcesDir="$demoDataDir/resources/"
if [ ! -d "$resourcesDir" ]; then
    echo -e "\033[34m[5/huregeil]\033[0m Downloading the MIMIC-III demo config files from original repo"
    sudo mkdir -p $resourcesDir
    outputVariableMap="$resourcesDir/itemid_to_variable_map.csv"
    outputDefinitions="$resourcesDir/hcup_ccs_2015_definitions.yaml"
    sudo wget "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/itemid_to_variable_map.csv" -O "$outputVariableMap"
    sudo wget "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml" -O "$outputDefinitions"
else
    echo -e "\033[34m[5/huregeil]\033[0m MIMIC-III demo config files already downloaded"
fi

# Resolve script and target locations
downloadScript="$testFolder/etc/benchmark_scripts/download_ci_dataset.py"
controlDatasetDir="$testFolder/data/control-dataset"

# Control the dataset from Google Drive
if is_empty "$controlDatasetDir"; then
    echo -e "\033[34m[6/huregeil]\033[0m Downloading the readily preprocessed control dataset"
    sudo -E env PATH="$PATH" PYTHONPATH="$PYTHONPATH" python "$downloadScript"
else
    echo -e "\033[34m[6/huregeil]\033[0m Control dataset already exists"
fi
