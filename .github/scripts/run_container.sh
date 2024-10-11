#!/bin/bash

# Assign parameters to variables for clarity and better control
DOCKER_VOLUME_MOUNTS="${1}"
BRANCH_NAME="${2}"
COMMAND="${3}"
BASH_RESULTS="${4}"
OUTPUT_FILENAME="${5}"

BLUE="\033[0;34m"
RESET="\033[0m"

echo -e ""
echo -e "${BLUE}=========== .github/scripts/run_container.sh ================"
echo -e "${BLUE}- Docker volume: ${RESET}$DOCKER_VOLUME_MOUNTS"
echo -e "${BLUE}- Branch name: ${RESET}$BRANCH_NAME"
echo -e "${BLUE}- Command: ${RESET}$COMMAND"
echo -e "${BLUE}- Bash results: ${RESET}$BASH_RESULTS"
echo -e "${BLUE}- Output filename: ${RESET}$OUTPUT_FILENAME"
echo -e ""

set -o pipefail

echo -e "${BLUE}Running command:${REST}\n \
    docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "$COMMAND 2>&1" \
    | tee $BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}\n"

# Running the pytest command inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "$COMMAND 2>&1" \
    | tee $BASH_RESULTS/$OUTPUT_FILENAME.txt

# Capture the exit status of the pytest command
test_status=$?

# Printing and handling the exit status
echo -e "${BLUE}---------- Exit status: $test_status--------------------------------${RESET}"
echo -e "Log artifact located at: \
    $BASH_RESULTS/$OUTPUT_FILENAME.txt"
if [ $test_status -ne 0 ]; then
    echo "The command failed."
    exit $test_status
elif [ ! -f "$BASH_RESULTS/$OUTPUT_FILENAME.txt" ]; then
    echo "Log artifact not created."
    exit $(( $test_status == 0 ? 1 : $test_status ))
fi
echo ""
