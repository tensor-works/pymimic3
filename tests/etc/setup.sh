# Check for wget and install if not present
if ! command -v wget &> /dev/null; then
    echo "wget could not be found. Attempting to install..."
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
    echo "wget is already installed."
fi

echo Setting up environment with OS type: $OSTYPE
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$( cd "$( dirname "$0" )" && pwd )"
fi

# Create the existing folder index
sourceUrl="https://physionet.org/files/mimiciii-demo/1.4/"
testFolder=$(dirname $(dirname $SCRIPT))

# Download the MIMIC-III demo dataset from the web
destinationDir="$testFolder/data/"
convertScript="$testFolder/etc/benchmark_scriptsconvert_columns.py"

if [ ! -d "$destinationDir/physionet.org" ]; then
    echo "Downloading the MIMIC-III demo dataset directory..."
    sudo wget -r -N -c -np $sourceUrl -P $destinationDir
    
    # Correcting defaults of the demo dataset
    echo "Correcting defaults of the demo dataset"
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $convertScript
fi

csvDir="$destinationDir/physionet.org/files/mimiciii-demo/1.4/"
resourcesDir="$csvDir/resources/"
if [ ! -d "$resourcesDir" ]; then
    sudo mkdir -p $resourcesDir
    outputVariableMap="$resourcesDir/itemid_to_variable_map.csv"
    outputDefinitions="$resourcesDir/hcup_ccs_2015_definitions.yaml"
    sudo wget "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/itemid_to_variable_map.csv" -O "$outputVariableMap"
    sudo wget "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml" -O "$outputDefinitions"
fi

generatedDir="$testFolder/data/mimic3benchmarks"
if [ ! -d "$generatedDir" ]; then
    echo "Downloading MIMIC-III benchmarks dataset from github"
    git clone "https://github.com/YerevaNN/mimic3-benchmarks.git" $generatedDir
fi

renameScript="$testFolder/etc/benchmark_scriptsrename_files.py"
revertSplitScript="$testFolder/etc/benchmark_scriptsrevert_split.py"
engineScript="$testFolder/etc/benchmark_scriptsengineer_data.py"
discretizerScript="$testFolder/etc/benchmark_scriptsdiscretize_data.py"

# Change into the MIMIC-III benchmarks directory
currentDir=$(pwd)
cd $generatedDir

# Checkout specific commit to keep this stable 
git fetch --all
echo "Switching to 2023 remote version at commit ea0314c7" 
git checkout ea0314c7cbd369f62e2237ace6f683740f867e3a > /dev/null 2>&1
extractedDir="$destinationDir/generated-benchmark/extracted/"
processedDir="$destinationDir/generated-benchmark/processed/"
engineeredDir="$destinationDir/generated-benchmark/engineered/"
discretizedDir="$destinationDir/generated-benchmark/discretized/"

if [ ! -d "$extractedDir" ]; then
    # Run the MIMIC-III benchmarks dataset processing
    echo "Extracting subject informations and timeseries data using original MIMIC-III github"
    python -m mimic3benchmark.scripts.extract_subjects $csvDir $extractedDir
    python -m mimic3benchmark.scripts.validate_events $extractedDir
    python -m mimic3benchmark.scripts.extract_episodes_from_subjects $extractedDir

    echo "Renaming files to include icustay_id in the filename"
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $renameScript
    python -m mimic3benchmark.scripts.split_train_and_test $extractedDir
fi

if [ ! -d "$processedDir/in-hospital-mortality" ] || [ ! -d "$processedDir/decompensation" ] || [ ! -d "$processedDir/length-of-stay" ] || [ ! -d "$processedDir/phenotyping" ] || [ ! -d "$processedDir/multitask" ]; then
    echo "Processing task data using the original MIMIC-III github"
    python -m mimic3benchmark.scripts.create_in_hospital_mortality $extractedDir "$processedDir/in-hospital-mortality/"
    python -m mimic3benchmark.scripts.create_decompensation $extractedDir "$processedDir/decompensation/"
    python -m mimic3benchmark.scripts.create_length_of_stay $extractedDir "$processedDir/length-of-stay/"
    python -m mimic3benchmark.scripts.create_phenotyping $extractedDir "$processedDir/phenotyping/"
    python -m mimic3benchmark.scripts.create_multitask $extractedDir "$processedDir/multitask/"
    echo "Reverting the dataset split from original MIMIC-III github"
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $revertSplitScript
fi

if [ ! -d "$engineeredDir/in-hospital-mortality" ] || [ ! -d "$engineeredDir/decompensation" ] || [ ! -d "$engineeredDir/length-of-stay" ] || [ ! -d "$engineeredDir/phenotyping" ]; then
    echo "Engineering task data using the original MIMIC-III github"
    engineScriptTarget="$generatedDir/engineer_data.py"
    cp $engineScript $engineScriptTarget
    python $engineScriptTarget
fi

if [ ! -d "$discretizedDir/in-hospital-mortality" ] || [ ! -d "$discretizedDir/decompensation" ] || [ ! -d "$discretizedDir/length-of-stay" ] || [ ! -d "$discretizedDir/phenotyping" ]; then
    echo "Discretizing task data using the original MIMIC-III github"
    discretizerScriptTarget="$generatedDir/discretize_data.py"
    cp $discretizerScript $discretizerScriptTarget
    sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH python $discretizerScriptTarget
fi
 
cd $currentDir

echo "Removing MIMIC-III benchmarks dataset from github"
sudo rm -rf $generatedDir


