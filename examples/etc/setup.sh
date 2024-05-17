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
exampleFolder=$(dirname $(dirname $SCRIPT))

# Download the MIMIC-III demo dataset from the web
destinationDir="$testFolder/data/"
convertScript="$testFolder/etc/convert_columns.py"

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