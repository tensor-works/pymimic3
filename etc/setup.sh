#!/bin/sh

# Create the environment variables
echo Setting up environment with OS type: $OSTYPE
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$( cd "$( dirname "$0" )" && pwd )"
fi

export CONFIG=`dirname $SCRIPT`
export WORKINGDIR=`dirname $CONFIG`
export FRONTEND=$WORKINGDIR/frontend
export MODEL=$WORKINGDIR/models
export TESTS=$WORKINGDIR/tests

# Create the mimic3-benchmark directory if not existing
envFile="$WORKINGDIR/.devcontainer/linux-gnu.yml"

source ~/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
if conda env list | grep -q "mimic3"; then
    echo "(mimic3) conda env already exists. Delete if you want to recreate."
else
    # Check if libmamba is already set as the solver
    if conda config --show | grep -q "solver: libmamba"; then
        echo "libmamba solver is already in use."
    else
        echo "Updating conda"
        conda update -n base -c defaults conda # 24.3.0 at time of creation
        echo "Installing Libmamba solver"
        conda install -n base conda-libmamba-solver
        # Set libmamba as the default solver
        conda config --set solver libmamba
    fi
    echo "Creating (mimic3) conda env" 
    conda env create -f $envFile
fi

conda activate mimic3

# Fetch settings files from original mimic3 directory
destinationDir="$CONFIG/mimic3-benchmark"
if [ ! -d "$destinationDir" ]; then
    sudo mkdir -p "$destinationDir"
fi
outputVariableMap="$destinationDir/itemid_to_variable_map.csv"
outputDefinitions="$destinationDir/hcup_ccs_2015_definitions.yaml"

if [ ! -f "$outputVariableMap" ]; then
    echo "Downloading itemid_to_variable_map.csv"
    sudo wget "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/itemid_to_variable_map.csv" -O "$outputVariableMap"
fi
if [ ! -f "$outputDefinitions" ]; then
    echo "Downloading hcup_ccs_2015_definitions.yaml"
    sudo wget "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml" -O "$outputDefinitions"
fi

# Creates redundancy in python path when sourced out of integreated shell but makes sure works for external shell aswell
echo "Setting PYTHONPATH"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$WORKINGDIR/src" != "$PYTHONPATH" ]]; then
        export PYTHONPATH=$PYTHONPATH:$WORKINGDIR/src
    fi
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
    if [[ "$WORKINGDIR/src" != "$PYTHONPATH" ]]; then
        export PYTHONPATH=$PYTHONPATH:$WORKINGDIR\\src
    fi  
fi

# Create the .env file
echo "Creating .env file"
if ! python -m pip list | grep -Fq python-dotenv; then
    python -m pip install python-dotenv
fi

dotenv -f ${WORKINGDIR}/.env set WORKINGDIR ${WORKINGDIR} 
dotenv -f ${WORKINGDIR}/.env set CONFIG ${CONFIG} 
dotenv -f ${WORKINGDIR}/.env set MODEL ${MODEL} 
dotenv -f ${WORKINGDIR}/.env set PYTHONPATH ${PYTHONPATH} 
dotenv -f ${WORKINGDIR}/.env set TESTS ${TESTS}

source "${CONFIG}/envvars.sh"


