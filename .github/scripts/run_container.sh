#!/bin/bash

# Assign parameters to variables for clarity and better control
DOCKER_VOLUME_MOUNTS="${1}"
BRANCH_NAME="${2}"
COMMAND="${3}"
CONTAINER_BASH_RESULTS="${4}"
BASH_RESULTS="${5}"
OUTPUT_FILENAME="${6}"

echo "Docker volume: $DOCKER_VOLUME_MOUNTS"
echo "Branch name: $BRANCH_NAME"
echo "Command: $COMMAND"
echo "Contianer bash results: $CONTAINER_BASH_RESULTS"
echo "Bash results: $BASH_RESULTS"
echo "Output filename: $OUTPUT_FILENAME"

# Running the pytest command inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "$COMMAND 2>&1" \
    | tee $BASH_RESULTS/$OUTPUT_FILENAME.txt

# Capture the exit status of the pytest command
test_status=$?

# Printing and handling the exit status
echo "Exit status: $test_status"
if [ $test_status -ne 0 ]; then
    echo "The command failed."
    exit $test_status
elif [ ! -f "$BASH_RESULTS/$OUTPUT_FILENAME.txt" ]; then
    echo "Log artifact not created."
    exit $(( $test_status == 0 ? 1 : $test_status ))
fi
