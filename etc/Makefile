# Makefile

# Default target that depends on other targets
all: env_vars config_env env_downloads env_conda

# Target for env_vars.sh
env_vars:
	@echo "Sourcing env_vars.sh..."
	@bash ./setup/env_vars.sh

# Target for env_downloads.sh
env_downloads:
	@echo "Sourcing env_downloads.sh..."
	@bash ./setup/env_downloads.sh

# Target for env_conda.sh
env_conda:
	@echo "Sourcing env_conda.sh..."
	@bash ./setup/env_conda.sh

.PHONY: all env_vars config_env env_downloads env_conda
