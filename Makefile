# Makefile
.ONESHELL:
SHELL := /bin/bash

# Default target that depends on other targets
all: .env load_env setup_config_files setup_conda_env setup_tests

.PHONY: .env
# Target for setup_env_vars.sh
.env:
	@echo "Running setup_env_vars.sh..."
	@bash ./etc/setup/setup_env_vars.sh; \

.PHONY: config_env setup_config_files setup_conda_env setup_tests

# Function to load .env and export variables
load_env: 
	@echo "Loading environment variables..."
	$(eval include .env)
	$(eval export)

# Target for setup_config_file.sh
setup_config_files: 
	@echo "DEBUG: Environment variables:"
	@echo "Running setup_config_files.sh..."
	@bash ./etc/setup/setup_config_files.sh

# Target for setup_config_file.sh
setup_conda_env:
ifeq ($(IS_DOCKER), 1)
	@echo "Skipping setup_conda_env step because IS_DOCKER is set to 1."
else
	@echo "Running setup_conda_env.sh..."
	@bash ./etc/setup/setup_conda_env.sh
endif

setup_tests:
ifeq ($(testing), true)
ifeq ($(IS_DOCKER), 1)
	@echo "Running setup_ci_tests.sh..."
	@bash ./etc/setup/setup_ci_tests.sh
else
	@echo "Running setup_tests.sh..."
	@bash ./etc/setup/setup_tests.sh
endif
else
	@echo "Skipping setup_tests step because testing is not enabled. Set testing=true to setup tests."
endif

