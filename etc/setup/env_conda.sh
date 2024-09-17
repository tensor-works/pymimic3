#!/bin/sh

# Create the mimic3-benchmark directory if not existing
echo "In etc/setup/env_conda.sh"

echo -e "\033[34m[1/6]\033[0m Detected OS type: $OSTYPE"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$( cd "$( dirname "$0" )" && pwd )"
fi

echo -e "\033[34m[2/6]\033[0m Sourcing environment variables"
source $(dirname $SCRIPT)/../../.env

envFile="$WORKINGDIR/.devcontainer/linux-gnu.yml"

eval "$(conda shell.bash hook)"
if conda env list | grep -q "mimic3"; then
    echo -e "\033[34m[3/6]\033[0m mimic3 conda env already exists. Delete if you want to recreate."
else
    # Check if libmamba is already set as the solver
    if conda config --show | grep -q "solver: libmamba"; then
        echo -e "\033[34m[3/6]\033[0m Libmamba solver is already in use."
    else
        echo -e "\033[34m[3/6]\033[0m Updating conda"
        conda update -n base -c defaults conda -y # 24.3.0 at time of creation
        echo -e "\033[34m[4/6]\033[0m Installing Libmamba solver"
        conda install -yn base conda-libmamba-solver
        # Set libmamba as the default solver
        conda config --set solver libmamba
    fi
    echo "\033[34m[5/6]\033[0m Creating mimic3 conda env from file $envFile" 
    conda env create -yf $envFile
fi

echo -e "\033[34m[6/6]\033[0m Activating mimic3 conda environment"
conda activate mimic3

export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

python -m dotenv -f ${WORKINGDIR}/.env set CUDNN_PATH ${CUDNN_PATH}
python -m dotenv -f ${WORKINGDIR}/.env set LD_LIBRARY_PATH ${LD_LIBRARY_PATH}