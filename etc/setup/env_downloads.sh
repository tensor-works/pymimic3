#!/bin/sh

# Need env_vars.sh
echo "In etc/setup/env_downloads.sh"

echo -e "\033[34m[1/5]\033[0m Detected OS type: $OSTYPE"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$( cd "$( dirname "$0" )" && pwd )"
fi

echo -e "\033[34m[2/5]\033[0m Sourcing environment variables"
source $(dirname $SCRIPT)/../../.env

echo -e "\033[34m[3/5]\033[0m Creating benchmark download dir at $CONFIG/mimic3-benchmark"
# Fetch settings files from original mimic3 directory
destinationDir="$CONFIG/mimic3-benchmark"
if [ ! -d "$destinationDir" ]; then
    mkdir -p "$destinationDir"
fi
outputVariableMap="$destinationDir/itemid_to_variable_map.csv"
outputDefinitions="$destinationDir/hcup_ccs_2015_definitions.yaml"

if [ ! -f "$outputVariableMap" ]; then
    echo -e "\033[34m[4/5]\033[0m Downloading itemid_to_variable_map.csv"
    wget "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/itemid_to_variable_map.csv" -O "$outputVariableMap"
else
    echo -e "\033[34m[4/5]\033[0m itemid_to_variable_map.csv already exists"
fi
if [ ! -f "$outputDefinitions" ]; then
    echo -e "\033[34m[5/5]\033[0m Downloading hcup_ccs_2015_definitions.yaml"
    wget "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml" -O "$outputDefinitions"
else
    echo -e "\033[34m[5/5]\033[0m hcup_ccs_2015_definitions.yaml already exists"
fi
