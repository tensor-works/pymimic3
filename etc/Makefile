# Makefile

# Default target that depends on other targets
all: env_vars config_env setup_config_file setup_conda_env setup_tests

# Target for setup_env_vars.sh
env_vars:
	@echo "Sourcing setup_env_vars.sh..."
	@bash ./setup/setup_env_vars.sh

# Target for setup_config_file.sh
setup_config_files:
	@echo "Sourcing setup_config_files.sh..."
	@bash ./setup/setup_config_files.sh

# Target for setup_config_file.sh
setup_conda_env:
ifeq ($(IS_DOCKER), 1)
	@echo "Skipping setup_conda_env step because IS_DOCKER is set to 1."
else
	@echo "Sourcing setup_conda_env.sh..."
	@bash ./setup/setup_conda_env.sh
endif

setup_tests:
ifeq ($(testing), true)
	@echo "Sourcing setup_config_file.sh..."
	@bash ./setup/setup_tests.sh
else
	@echo "Skipping setup_tests step because. Set testing=true to setup tests."
endif

.PHONY: all env_vars config_env setup_config_file setup_conda_env setup_tests
