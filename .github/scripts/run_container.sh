#!/bin/bash

# Assign parameters to variables for clarity and better control
DOCKER_VOLUME_MOUNTS="${1}"
BRANCH_NAME="${2}"
COMMAND="${3}"
BASH_RESULTS="${4}"
OUTPUT_FILENAME="${5}"

RED='\033[31;1m'
BLUE='\033[34;1m'
LIGHT_BLUE='\033[94m'
GREEN='\033[32;1m'
RESET='\033[0m'

# For echoing
FORMATTED_MOUNTS=$(echo "$DOCKER_VOLUME_MOUNTS" | sed "s/ -v /\n  \\${LIGHTBLUE} /g")

echo -e ""
echo -e "${BLUE}=========== .github/scripts/run_container.sh ================"
echo -e "${BLUE}- Docker volume: ${LIGHT_BLUE}$FORMATTED_MOUNTS"
echo -e "${BLUE}- Branch name: ${LIGHT_BLUE}$BRANCH_NAME"
echo -e "${BLUE}- Command: ${LIGHT_BLUE}$COMMAND"
echo -e "${BLUE}- Bash results: ${LIGHT_BLUE}$BASH_RESULTS"
echo -e "${BLUE}- Output filename: ${LIGHT_BLUE}$OUTPUT_FILENAME${RESET}"
echo -e "${BLUE}----------- Artifacts and logs ------------------------------"
echo -e "${BLUE}Log artifact located at:${LIGHT_BLUE} \


set -o pipefail

echo "::group::Container command"
echo -e "${BLUE}Running command:${REST}\n \
    docker run $FORMATTED_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic \"$COMMAND 2>&1\" \
    | tee $BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}\n"
echo "::endgroup::"

# Running the pytest command inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "$COMMAND 2>&1" \
    | tee $BASH_RESULTS/$OUTPUT_FILENAME.txt

# Capture the exit status of the pytest command
test_status=$?

# Printing and handling the exit status
echo -e "${BLUE}---------- Exit status: $test_status --------------------------------"
    $BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}"
if [ $test_status -ne 0 ]; then
    echo "${RED}The command failed.${RESET}"
    exit $test_status
elif [ ! -f "$BASH_RESULTS/$OUTPUT_FILENAME.txt" ]; then
    echo "${RED}Log artifact not created. Expected location:\n \
        $BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}"
    exit $(( $test_status == 0 ? 1 : $test_status ))
fi
echo ""
