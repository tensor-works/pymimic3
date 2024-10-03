# !bin/bash

echo "In etc/setup/setup_tests.sh"

# Check for wget and install if not present
if ! command -v wget &> /dev/null; then
    echo -e "\033[34m[1/huregeil]\033[0m wget could not be found. Attempting to install..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install wget -y
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install wget
        else
            echo "Homebrew not found. Please install Homebrew and rerun the script or manually install wget."
        fi
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        echo "Please install wget manually using your package manager or visit https://www.gnu.org/software/wget/"
    else
        echo "Unsupported OS for automatic wget installation. Please use setup.ps1."
    fi
else
    echo -e "\033[34m[1/huregeil]\033[0m wget is already installed."
fi

# Check OS-type and resolve SCRIPT directory
echo -e "\033[34m[2/huregeil]\033[0m Setting up environment with OS type: $OSTYPE"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$( cd "$( dirname "$0" )" && pwd )"
fi

# Create the existing folder index
sourceUrl="https://physionet.org/files/mimiciii-demo/1.4/"
testFolder="$(dirname $(dirname $(dirname $SCRIPT)))/tests"

# Download the MIMIC-III demo dataset from the web
destinationDir="$testFolder/data/"
convertScript="$testFolder/etc/benchmark_scripts/convert_columns.py"
csvDir="$destinationDir/mimiciii-demo/"

if [ ! -d "$destinationDir/mimiciii-demo" ]; then
    echo -e "\033[34m[3/huregeil]\033[0m Downloading the MIMIC-III demo dataset directory"
    sudo wget -r -N -c -np $sourceUrl -P $destinationDir
    
    # Correcting defaults of the demo dataset
    echo -e "\033[34m[4/huregeil]\033[0m Correcting headers of the MIMIC-III demo dataset"
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $convertScript
    sudo mkdir $csvDir
    origCsvDir="$destinationDir/physionet.org/files/mimiciii-demo/1.4/"
    sudo cp $origCsvDir/* $csvDir
else
    echo -e "\033[34m[3/huregeil]\033[0m MIMIC-III demo dataset directory already exists"
    echo -e "\033[34m[4/huregeil]\033[0m Header already corrected"
fi
resourcesDir="$csvDir/resources/"
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

controlRepositorDir="$testFolder/data/mimic3benchmarks"
if [ ! -d "$controlRepositorDir" ]; then
    echo -e "\033[34m[5/huregeil]\033[0m Downloading original MIMIC-III benchmarks code from github"
    git clone "https://github.com/YerevaNN/mimic3-benchmarks.git" $controlRepositorDir
else
    echo -e "\033[34m[5/huregeil]\033[0m Original MIMIC-III benchmarks code already downloaded"
fi

renameScript="$testFolder/etc/benchmark_scripts/rename_files.py"
extractScript="$testFolder/etc/benchmark_scripts/run_extraction.py"
processTaskScript="$testFolder/etc/benchmark_scripts/run_process_tasks.py"
revertSplitScript="$testFolder/etc/benchmark_scripts/revert_split.py"
engineScript="$testFolder/etc/benchmark_scripts/engineer_data.py"
discretizerScript="$testFolder/etc/benchmark_scripts/discretize_data.py"

# Change into the MIMIC-III benchmarks directory
currentDir=$(pwd)
cd $controlRepositorDir

# Checkout specific commit to keep this stable 
git fetch --all
echo "Switching to 2023 remote version at commit ea0314c7" 
git checkout ea0314c7cbd369f62e2237ace6f683740f867e3a > /dev/null 2>&1
extractedDir="$destinationDir/control-dataset/extracted/"
extractedTestDir="$destinationDir/control-dataset/extracted/test/"
processedDir="$destinationDir/control-dataset/processed/"
engineeredDir="$destinationDir/control-dataset/engineered/"
discretizedDir="$destinationDir/control-dataset/discretized/"

# Extract the data
if [ ! -d "$extractedDir" ]; then
    # Run the MIMIC-III benchmarks dataset processing
    echo -e "\033[34m[6/huregeil]\033[0m Extracting subject informations and timeseries data using original MIMIC-III github"
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $extractScript $csvDir $extractedDir $controlRepositorDir

    echo -e "\033[34m[7/huregeil]\033[0m Renaming episode files to include ICUSTAY_ID in the filename"
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $renameScript

else
    echo -e "\033[34m[6/huregeil]\033[0m Control dataset already extracted"
    echo -e "\033[34m[7/huregeil]\033[0m Extracted control dataset already renamed"
fi

# Split the data
if [ ! -d "$extractedTestDir" ]; then
    echo -e "\033[34m[8/huregeil]\033[0m Split the dataset into training and testing sets"
    python -m mimic3benchmark.scripts.split_train_and_test $extractedDir
else
    echo -e "\033[34m[8/huregeil]\033[0m Extracted control dataset already split"
fi

# Process the data
if [ ! -d "$processedDir/in-hospital-mortality" ] || [ ! -d "$processedDir/decompensation" ] || [ ! -d "$processedDir/length-of-stay" ] || [ ! -d "$processedDir/phenotyping" ] || [ ! -d "$processedDir/multitask" ]; then
    echo -e "\033[34m[9/huregeil]\033[0m Processing task data using the original MIMIC-III github"
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $processTaskScript $extractedDir $processedDir $controlRepositorDir

    echo -e "\033[34m[10/huregeil]\033[0m Reverting the dataset split from original MIMIC-III github"
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $revertSplitScript
else
    echo -e "\033[34m[9/huregeil]\033[0m Processed task data already exists"
    echo -e "\033[34m[10/huregeil]\033[0m Dataset split already reverted"
fi

# Engineer the data
if [ ! -d "$engineeredDir/in-hospital-mortality" ] || [ ! -d "$engineeredDir/decompensation" ] || [ ! -d "$engineeredDir/length-of-stay" ] || [ ! -d "$engineeredDir/phenotyping" ]; then
    echo -e "\033[34m[11/huregeil]\033[0m Engineering task data using the original MIMIC-III github"
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $engineScript $processedDir $engineeredDir $controlRepositorDir
else
    echo -e "\033[34m[11/huregeil]\033[0m Engineered task data already exists"
fi

# Discretize the data
if [ ! -d "$discretizedDir/in-hospital-mortality" ] || [ ! -d "$discretizedDir/decompensation" ] || [ ! -d "$discretizedDir/length-of-stay" ] || [ ! -d "$discretizedDir/phenotyping" ]; then
    echo -e "\033[34m[12/huregeil]\033[0m Discretizing task data using the original MIMIC-III github"
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $discretizerScript $processedDir $discretizedDir $controlRepositorDir
else
    echo -e "\033[34m[12/huregeil]\033[0m  Discretized task data already exists"
fi
 
cd $currentDir

echo -e "\033[34m[12/huregeil]\033[0m Removing MIMIC-III github repository from local machine"
sudo rm -rf $controlRepositorDir


