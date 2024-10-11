#!/bin/bash

# Assign parameters to variables for clarity and better control
DOCKER_VOLUME_MOUNTS="${1}"
BRANCH_NAME="${2}"
CONTAINER_PYTEST_RESULTS="${3}"
PYTEST_MODULE_PATH="${4}"
BASH_RESULTS="${5}"
PYTEST_RESULTS="${6}"
OUTPUT_FILENAME="${7}"

echo ""
echo "${BLUE}=========== .github/scripts/run_container.sh ================"
echo "${BLUE}- Docker volume: ${RESET}$DOCKER_VOLUME_MOUNTS"
echo "${BLUE}- Branch name: ${RESET}$BRANCH_NAME"
echo "${BLUE}- Container pytest results: ${RESET}$CONTAIER_PYTEST_RESULTS"
echo "${BLUE}- Pytest module path: ${RESET}$PYTEST_MODULE_PATH"
echo "${BLUE}- Bash results: ${RESET}$BASH_RESULTS"
echo "${BLUE}- Pytest results: ${RESET}$PYTEST_RESULTS"
echo "${BLUE}- Output filename: ${RESET}$OUTPUT_FILENAME"
echo ""

set -o pipefail

echo -e "${BLUE}Running command:${REST}\n \
    docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "pytest --no-cleanup --junitxml=$CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml \
    -v $PYTEST_MODULE_PATH 2>&1 && $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.html" \
    | tee $BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}\n"

# Running the pytest command inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "pytest --no-cleanup --junitxml=$CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml \
    -v $PYTEST_MODULE_PATH 2>&1" \
    | tee $BASH_RESULTS/$OUTPUT_FILENAME.txt

# Capture the exit status of the pytest command
test_status=$?

echo -e "${BLUE}docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "python -m tests.pytest_utils.reporting $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml \
    2>&1" | tee -a $BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}\n"

# Run the reporting tool inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "python -m tests.pytest_utils.reporting $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml \
    2>&1" | tee -a $BASH_RESULTS/$OUTPUT_FILENAME.txt

# Printing and handling the exit status
echo -e "${BLUE}---------- Exit status: $test_status--------------------------------${RESET}"
echo -e "${BLUE}Log artifact located at:\n$BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}"
echo -e "${BLUE}Pytest artfact created at:\n$PYTEST_RESULTS/$OUTPUT_FILENAME.xml${RESET}"
echo -e "${BLUE}Pytest artfact created at:\n$PYTEST_RESULTS/$OUTPUT_FILENAME.xml${RESET}"

if [ $test_status -ne 0 ]; then
    echo "The command failed."
    exit $test_status
elif [ ! -f "$BASH_RESULTS/$OUTPUT_FILENAME.txt" ]; then
    echo "Log artifact not created."
    exit $(( $test_status == 0 ? 1 : $test_status ))
elif [ ! -f "$PYTEST_RESULTS/$OUTPUT_FILENAME.xml" ]; then
    echo "Pytest artifact not created."
    exit $(( $test_status == 0 ? 1 : $test_status ))
fi
