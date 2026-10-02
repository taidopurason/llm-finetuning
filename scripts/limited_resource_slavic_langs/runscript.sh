#!/bin/bash -e

if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
fi
sleep 2

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Set MIOpen cache to a temporary folder.
if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

# Report affinity
echo "Rank $SLURM_PROCID --> $(taskset -p $$)"

# Start conda environment inside the container
$WITH_CONDA || true
# TODO: Change this to the correct path for the limited-llms venv
source ./venvs/limited-llms/bin/activate

# Set interfaces to be used by RCCL.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_DEFAULT_CQ_SIZE=262144    # default 131072

export RANK=$SLURM_PROCID

echo "HF_HOME: $HF_HOME"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"

"$@"


