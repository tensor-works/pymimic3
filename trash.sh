DOCKER_VOLUME_MOUNTS="-v /home/amadou/Data/.cache//.github-action-cache/control-dataset:/workdir/tests/data/control-dataset -v /home/amadou/Data/.cache//.github-action-cache/mimiciii-demo:/workdir/tests/data/mimiciii-demo -v /home/amadou/Data/.cache//.github-action-cache/semitemp:/workdir/tests/data/semitemp -v /home/amadou/Data/.cache//.github-action-cache/bash-results:/workdir/tests/data/bash-results -v /home/amadou/Data/.cache//.github-action-cache/pytest-results:/workdir/tests/data/pytest-results"
BRANCH_NAME=feature_workflow_pytests
CONTAINER_PYTEST_RESULTS=""
PYTEST_MODULE_PATH="tests/test_metrics"
BASH_RESULTS="/home/amadou/Data/.cache//.github-action-cache/bash-results/e6de761f3bc34de6b35bfd9be2c426dd1213d61f"
PYTEST_RESULTS="/home/amadou/Data/.cache//.github-action-cache/pytest-results/e6de761f3bc34de6b35bfd9be2c426dd1213d61f"
OUTPUT_FILENAME="test-metrics"

# Running the pytest command inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "pytest --no-cleanup --junitxml=$CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml \
    -v $PYTEST_MODULE_PATH 2>&1" \
    | tee $BASH_RESULTS/$OUTPUT_FILENAME.txt

# Capture the exit status of the pytest command
test_status=$?

# Run the reporting tool inside a Docker container
docker run $DOCKER_VOLUME_MOUNTS \
    tensorpod/pymimic3:$BRANCH_NAME \
    bash -ic "python -m tests.pytest_utils.reporting $CONTAINER_PYTEST_RESULTS/$OUTPUT_FILENAME.xml \
    2>&1" | tee -a $BASH_RESULTS/$OUTPUT_FILENAME.txt

# Printing and handling the exit status
echo "Exit status: $test_status"
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