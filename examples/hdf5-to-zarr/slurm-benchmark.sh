#!/bin/bash
#SBATCH --job-name=Benchmark
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=run-%j.out
#SBATCH --error=run-%j.err
#SBATCH --partition=main
#SBATCH --mem=24GB
#SBATCH --constraint infiniband
#SBATCH --time=48:00:00
if [ -z "$1" ]; then
        echo "[ERROR] [$(date -Iseconds)] [$$] SLURM account not provided."
        exit 1
fi
#SBATCH --account="$1"

set -eu

function display_memory_usage() {
        set +eu
        echo -n "[INFO] [$(date -Iseconds)] [$$] Max memory usage in bytes: "
        cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.max_usage_in_bytes
        echo
}

trap display_memory_usage EXIT

START=$(date +%s)
STARTDATE=$(date -Iseconds)
echo "[INFO] [$STARTDATE] [$$] Starting SLURM job $SLURM_JOB_ID"
echo "[INFO] [$STARTDATE] [$$] Running in $(hostname -s)"
echo "[INFO] [$STARTDATE] [$$] Working directory: $(pwd)"

### PUT YOUR CODE IN THIS SECTION
export PIXI_CACHE_DIR=/tungstenfs/scratch/gmicro_share/_software/pixi/cache/

# Run training in plant-seg environment
pixi run python benchmark.py

pixi run python benchmark_FS.py

pixi run python benchmark_ZIP.py

### END OF PUT YOUR CODE IN THIS SECTION

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time \(seconds\) : $(( $END-$START ))"
