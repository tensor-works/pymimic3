export SCRIPT=$(realpath $0)  
export WORKINGDIR=$(dirname $SCRIPT)

echo "Sourcing .env file"
echo "WORKINGDIR=${WORKINGDIR}"
set -a
source ${WORKINGDIR}/.env
set +a
