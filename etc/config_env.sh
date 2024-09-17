#!/bin/sh

# Need env_vars.sh
echo "In etc/config-env.sh"

echo -e "\033[34m[1/2]\033[0m Detected OS type: $OSTYPE"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$( cd "$( dirname "$0" )" && pwd )"
fi

export WORKINGDIR=$(dirname$(dirname $SCRIPT))

echo -e "\033[34m[2/2]\033[0m Sourcing .env file at ${WORKINGDIR}/.env"
set -a
source ${WORKINGDIR}/.env
set +a
