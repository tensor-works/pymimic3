#!/bin/bash

source ./.github/scripts/functions.sh

# Assign parameters to variables for clarity and better control
DOCKER_VOLUME_MOUNTS="${1}"
BRANCH_NAME="${2}"
CONTAINER_PYTEST_RESULTS="${3}"
PYTEST_MODULE_PATH="${4}"
BASH_RESULTS="${5}"
PYTEST_RESULTS="${6}"
OUTPUT_FILENAME="${7}"

RED='\033[31;1m'
BLUE='\033[34;1m'
LIGHT_BLUE='\033[94m'
GREEN='\033[32;1m'
RESET='\033[0m'

# Check if network exists and set network flag accordingly
NETWORK_FLAG=""
RUNNER_NETWORK=$(docker network ls --format '{{.Name}}' | grep '.*runner_network$' | head -n 1)
if [ ! -z "$RUNNER_NETWORK" ]; then
    NETWORK_FLAG="--network $RUNNER_NETWORK"
    echo -e "${BLUE}Found runner network: ${LIGHT_BLUE}$RUNNER_NETWORK${RESET}"
else
    echo -e "${BLUE}Warning: runner network not found, container will use default network${RESET}"
fi

# One db per action pipeline
GITHUB_SHA=$(get_sha "$BASH_RESULTS")

# For echoing
FORMATTED_MOUNTS=$(echo "$DOCKER_VOLUME_MOUNTS" | sed "s/ -v /\n  \\${LIGHTBLUE} /g")

echo -e ""
echo -e "${BLUE}=========== .github/scripts/run_pytest_container.sh ================"
echo -e "${BLUE}- Docker volume: ${LIGHT_BLUE}$FORMATTED_MOUNTS"
echo -e "${BLUE}- Branch name: \n  ${LIGHT_BLUE}$BRANCH_NAME"
echo -e "${BLUE}- Container pytest results: \n  ${LIGHT_BLUE}$CONTAINER_PYTEST_RESULTS"
echo -e "${BLUE}- Pytest module path: \n  ${LIGHT_BLUE}$PYTEST_MODULE_PATH"
echo -e "${BLUE}- Bash results: \n  ${LIGHT_BLUE}$BASH_RESULTS"
echo -e "${BLUE}- Pytest results: \n  ${LIGHT_BLUE}$PYTEST_RESULTS"
echo -e "${BLUE}- Output filename: \n  ${LIGHT_BLUE}$OUTPUT_FILENAME${RESET}"
echo -e "${BLUE}- Action SHA: \n  ${LIGHT_BLUE}$GITHUB_SHA${RESET}"
echo -e "${BLUE}----------- Artifacts and logs -------------------------------------"
echo -e "${BLUE}Log artifact located at:\n  ${LIGHT_BLUE}$BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}"
echo -e "${BLUE}Pytest junit artfact created at:\n  ${LIGHT_BLUE}$PYTEST_RESULTS/$OUTPUT_FILENAME.xml${RESET}"
echo -e "${BLUE}Pytest html artfact created at:\n  ${LIGHT_BLUE}$PYTEST_RESULTS/$OUTPUT_FILENAME.html${RESET}"


set -o pipefail

echo -e "::group::${BLUE}Pytest container command${REST}"
echo -e "${BLUE}Running command:${REST}\n \
docker run $FORMATTED_MOUNTS\n \
$NETWORK_FLAG \
-e MONGODB_NAME="$GITHUB_SHA" \
tensorpod/pymimic3:$BRANCH_NAME\n \
bash -ic \"pytest --no-cleanup --junitxml=$CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml\n \
-v $PYTEST_MODULE_PATH 2>&1\n \
&& junit2html $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.html\n\" \
| tee $BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}\n"
echo "::endgroup::"

# Running the pytest command inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    $NETWORK_FLAG \
    -e MONGODB_NAME="$GITHUB_SHA" \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "pytest --no-cleanup --junitxml=$CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml \
    -v $PYTEST_MODULE_PATH \
    && junit2html $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.html 2>&1" \
    | tee $BASH_RESULTS/$OUTPUT_FILENAME.txt

# Capture the exit status of the pytest command
test_status=$?

echo "::group::Pytest summary command"
echo -e "${BLUE}docker run $DOCKER_VOLUME_MOUNTS \    
    $NETWORK_FLAG \
    -e MONGODB_NAME="$GITHUB_SHA" \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic \"python -m tests.pytest_utils.reporting $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml \
    2>&1\" | tee -a $BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}\n"
echo "::endgroup::"

# Run the reporting tool inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    $NETWORK_FLAG \
    -e MONGODB_NAME="$GITHUB_SHA" \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "python -m tests.pytest_utils.reporting $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml \
    2>&1" | tee -a $BASH_RESULTS/$OUTPUT_FILENAME.txt

# Printing and handling the exit status
echo -e "${BLUE}---------- Exit status: $test_status--------------------------------${RESET}"
if [ $test_status -ne 0 ]; then
    echo -e "${BLUE}The command failed.${RESET}"
    exit $test_status
elif [ ! -f "$BASH_RESULTS/$OUTPUT_FILENAME.txt" ]; then
    echo -e "${RED}Log artifact not created. Expected location:\n \
        $BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}"
    exit $(( $test_status == 0 ? 1 : $test_status ))
elif [ ! -f "$PYTEST_RESULTS/$OUTPUT_FILENAME.xml" ]; then
    echo -e "${RED}Pytest artifact not created. Expected location:\n \
        $BASH_RESULTS/$OUTPUT_FILENAME.txt${RESET}"
    exit $(( $test_status == 0 ? 1 : $test_status ))
fi
