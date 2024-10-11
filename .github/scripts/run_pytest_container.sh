#!/bin/bash

# Assign parameters to variables for clarity and better control
DOCKER_VOLUME_MOUNTS="${1}"
BRANCH_NAME="${2}"
CONTAINER_PYTEST_RESULTS="${3}"
CONTAINER_BASH_RESULTS="${4}"
BASH_RESULTS="${5}"
PYTEST_RESULTS="${6}"
PYTEST_MODULE_PATH="${7}"
OUTPUT_FILENAME="${8}"

# Running the pytest command inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "pytest --no-cleanup --junitxml=$CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME \
    -v $PYTEST_MODULE_PATH 2>&1 \
    | tee $CONTAINER_BASH_RESULTS/${OUTPUT_FILENAME%.xml}.txt"

# Capture the exit status of the pytest command
test_status=$?

# Run the reporting tool inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "python -m tests.pytest_utils.reporting $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME \
    2>&1 | tee -a $CONTAINER_BASH_RESULTS/${OUTPUT_FILENAME%.xml}.txt"

# Printing and handling the exit status
echo "Exit status: $test_status"
if [ $test_status -ne 0 ]; then
    echo "The command failed."
    exit $test_status
elif [ ! -f "$BASH_RESULTS/${OUTPUT_FILENAME%.xml}.txt" ]; then
    echo "Log artifact not created."
    exit $(( $test_status == 0 ? 1 : $test_status ))
elif [ ! -f "$PYTEST_RESULTS/$OUTPUT_FILENAME" ]; then
    echo "Pytest artifact not created."
    exit $(( $test_status == 0 ? 1 : $test_status ))
fi
