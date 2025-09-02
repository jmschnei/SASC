#!/usr/bin/env bash

usage() { echo "Usage: $0 [-i <path to base enroot image>] [-o <path to output enroot image>]" 1>&2; exit 1; }


# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
enroot_image=""
tag=""

while getopts ":i:o:" o; do
    case "${o}" in
        i)
            if [ -f  ${OPTARG} ]; then
                enroot_image=${OPTARG}
            else
                echo "enroot_image=${enroot_image} is not a valid path."
                usage
            fi
            ;;
        o)
            tag=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${enroot_image}" ] || [ -z "${tag}" ]; then
    usage
fi

echo "enroot_image = ${enroot_image}"
echo "tag = ${tag}"

# CONTAINER_SAVE=/netscratch/$USER/enroot/sc-$tag.sqsh
MOUNTS=~/.ssh:/root/.ssh,"`pwd`":/ws,/netscratch:/netscratch

CONTAINER_ARGS="
    --container-image=$enroot_image \
    --container-save=$tag \
    --container-mounts=$MOUNTS
"

RESOURCES_ARGS="
    --partition RTXA6000 \
    --mem=120G
"

JOBNAME="build_sc_$(date '+%Y%m%d-%H%M%S')"
srun -K \
    --job-name=$JOBNAME \
    $RESOURCES_ARGS \
    $CONTAINER_ARGS \
    pip install -r /ws/requirements.txt
