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
	@bash ./config_env.sh; ./setup/setup_config_files.sh

# Target for setup_config_file.sh
setup_conda_env:
ifeq ($(IS_DOCKER), 1)
	@echo "Skipping setup_conda_env step because IS_DOCKER is set to 1."
else
	@echo "Sourcing setup_conda_env.sh..."
	@bash ./config_env.sh; ./setup/setup_conda_env.sh
endif

setup_tests:
ifeq ($(testing), true)
ifeq ($(IS_DOCKER), 1)
	@echo "Sourcing setup_ci_tests.sh..."
	@bash source ./config_env.sh; ./setup/setup_ci_tests.sh
else
	@echo "Sourcing setup_tests.sh..."
	@bash source ./config_env.sh; ./setup/setup_tests.sh
endif
else
	@echo "Skipping setup_tests step because testing is not enabled. Set testing=true to setup tests."
endif

.PHONY: all env_vars config_env setup_config_file setup_conda_env setup_tests
