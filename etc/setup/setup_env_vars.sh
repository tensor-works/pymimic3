#!/bin/sh

# Create the environment variables
echo "In etc/setup/setup_env_vars.sh"
echo -e "\033[34m[1/4]\033[0m Detected OS type: $OSTYPE"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$( cd "$( dirname "$0" )" && pwd )"
fi

export ETC=$(dirname $(dirname $SCRIPT))
export CONFIG=$ETC/config
export WORKINGDIR=$(dirname $(dirname $CONFIG))
export FRONTEND=$WORKINGDIR/frontend
export MODEL=$WORKINGDIR/models
export TESTS=$WORKINGDIR/tests
export EXAMPLES=$WORKINGDIR/examples

# Creates redundancy in python path when sourced out of integreated shell but makes sure works for external shell aswell
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$WORKINGDIR/src" != "$PYTHONPATH" ]]; then
        export PYTHONPATH=$PYTHONPATH:$WORKINGDIR/src
    fi
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
    if [[ "$WORKINGDIR/src" != "$PYTHONPATH" ]]; then
        export PYTHONPATH=$PYTHONPATH:$WORKINGDIR\\src
    fi  
fi
echo -e "\033[34m[2/4]\033[0m Setting PYTHONPATH=${PYTHONPATH}"

# Create the .env file
if [ -z "$(python -m pip list | grep python-dotenv)" ]; then
    echo -e "\033[34m[3/4]\033[0m Installing python-dotenv"
    python -m pip install python-dotenv[cli] --no-cache-dir
else
    echo -e "\033[34m[3/4]\033[0m Python-dotenv already installed"
fi

echo -e "\033[34m[4/4]\033[0m Exporting ${WORKINGDIR}/.env"
python -m dotenv -q never -f  ${WORKINGDIR}/.env set WORKINGDIR ${WORKINGDIR}  > /dev/null 2>&1
python -m dotenv -q never -f  ${WORKINGDIR}/.env set CONFIG ${CONFIG}   > /dev/null 2>&1
python -m dotenv -q never -f  ${WORKINGDIR}/.env set ETC ${ETC}   > /dev/null 2>&1
python -m dotenv -q never -f  ${WORKINGDIR}/.env set MODEL ${MODEL}   > /dev/null 2>&1
python -m dotenv -q never -f  ${WORKINGDIR}/.env set PYTHONPATH ${PYTHONPATH}   > /dev/null 2>&1
python -m dotenv -q never -f  ${WORKINGDIR}/.env set TESTS ${TESTS}  > /dev/null 2>&1
python -m dotenv -q never -f  ${WORKINGDIR}/.env set EXAMPLES ${EXAMPLES}  > /dev/null 2>&1

chmod 777 ${WORKINGDIR}/.env