#!/bin/bash

# Name of the project
NAME_PROJECT="singularity-hydrogym"
# Name of the job script
NAME_JOB_ARRAY_SCRIPT="job_array_batch_xp_singularity.slurm"

PATH_PARENT=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)
# Path to the project's content root directory
PATH_CONTENT_ROOT="$WORKDIR"/pycharm_remote_project/"$NAME_PROJECT"

# Path of the **default** python script to run
# shellcheck disable=SC2034
NAME_MOUNT_DIR="mount_dir"
PATH_CONTAINER_CONTENT_ROOT="/home/firedrake/$NAME_MOUNT_DIR/project_root"

NAME_CONTAINER="hydrogym-firedrake_nousernamespace_uid-1001_gid-1001_hostname-mecacpt80.sif"
PATH_CONTAINER="$PATH_CONTENT_ROOT"/singularity/images/"$NAME_CONTAINER"

# Firedrake writes to the cache directory that is replicated on the Singularity container
# this triggers an OSError: [Errno 28] No space left on device
PATH_HOME_DIR_HPC="/gpfs/users/hosseinkhanr"
PATH_CACHE_DIR_HPC="$PATH_HOME_DIR_HPC"/.cache

# Get the folder name from the command line, and the arguments to pass to the python script
while getopts 'n:p:a:' flag; do
  case "${flag}" in
  n) NAME_FOLDER_CONFIGS="${OPTARG}" ;;
  p) RELATIVE_PATH_PYTHON_SCRIPT="${OPTARG}" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done

# Check that the folder name was provided
if [ -z "$NAME_FOLDER_CONFIGS" ]; then
  echo Missing folder name -n option.
  exit
fi

PATH_FOLDER_CONFIGS="$PATH_CONTENT_ROOT"/configs/batch/"$NAME_FOLDER_CONFIGS"

echo "Config folder: $PATH_FOLDER_CONFIGS"
echo

# Get the number of yaml files in the folder PATH_FOLDER_CONFIGS
N_CONFIGS=$(find "$PATH_FOLDER_CONFIGS" -name "*.yaml" | wc -l)
echo "Number of configs: $N_CONFIGS"

# Create the name of the log directory with the current date and time
PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$NAME_FOLDER_CONFIGS"/$(date +"%Y-%m-%d_%H-%M-%S")

echo "Log directory: $PATH_LOG_DIR"
echo

# Create the log directory for the current config file
mkdir -p "$PATH_LOG_DIR"/"$CONFIG_FILE_NAME"

# Set the MLFlow tracking URI, export will make it available to the subprocesses
export MLFLOW_TRACKING_URI=file:"$PATH_CONTENT_ROOT"/data/mlruns

# Launch the job array script
echo "Launching $NAME_JOB_ARRAY_SCRIPT"
echo

# Transform the relative paths to absolute paths
# Note: The CONTAINER CONTENT ROOT is given instead !
PATH_PYTHON_SCRIPT="$PATH_CONTAINER_CONTENT_ROOT"/"$RELATIVE_PATH_PYTHON_SCRIPT"

echo RELATIVE_PATH_PYTHON_SCRIPT: "$RELATIVE_PATH_PYTHON_SCRIPT"
echo
echo PATH_PYTHON_SCRIPT: "$PATH_PYTHON_SCRIPT"
echo
echo "MLFlow tracking URI: $MLFLOW_TRACKING_URI"
echo

# Set defaults values for the sbatch options
# --- Number of CPUs per task ---
#S_BATCH_CPU_PER_TASK=8
S_BATCH_CPU_PER_TASK=4

# --- Time limit ---
S_BATCH_TIME=48:00:00
#S_BATCH_TIME=3:59:00
#S_BATCH_TIME=9:59:00
#S_BATCH_TIME=59:00:00
#S_BATCH_TIME=00:40:00

# --- Partition ---
#S_BATCH_PARTITION=cpu_short
#S_BATCH_PARTITION=cpu_med
S_BATCH_PARTITION=cpu_long  # (12 cores at 3.2 GHz), namely 48 cores per node

# --- Quality of service ---
#S_BATCH_QOS=qos_cpu-t3
#S_BATCH_QOS=qos_cpu-t4
#S_BATCH_QOS=qos_cpu-dev

# --- Account ---
S_BATCH_ACCOUNT=rl_for_dy+

# Ruche specific options
# --- Number of nodes ---
#S_BATCH_NODES=1

# --- Number of tasks ---
#S_BATCH_N_TASKS=1

# --- Number of tasks per node ---
S_BATCH_N_TASKS_PER_NODE=1

# --- Number of GPUs ---
S_BATCH_GPUS=0

# --- Memory per node ---
S_BATCH_MEM_PER_NODE="8G"

# Get last array ID
N_LAST_ARRAYID=$((N_CONFIGS - 1))

echo "sbatch options:"
echo "  --job-name=$NAME_FOLDER_CONFIGS"
echo "  --output=$PATH_LOG_DIR/job_array_launcher_%A_%a.out"
echo "  --error=$PATH_LOG_DIR/job_array_launcher_%A_%a.err"
echo "  --export=NAME_PROJECT=$NAME_PROJECT,PATH_PYTHON_SCRIPT=$PATH_PYTHON_SCRIPT,PATH_FOLDER_CONFIGS=$PATH_FOLDER_CONFIGS,WORKDIR=$WORKDIR,MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI,PATH_CONTAINER=$PATH_CONTAINER,PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,PATH_CONTAINER_CONTENT_ROOT=$PATH_CONTAINER_CONTENT_ROOT,PATH_CACHE_DIR_HPC=$PATH_CACHE_DIR_HPC"
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
echo "  --time=$S_BATCH_TIME"
echo "  --partition=$S_BATCH_PARTITION"
#echo "  --qos=$S_BATCH_QOS"
echo "  --account=$S_BATCH_ACCOUNT"
echo "  --mem=$S_BATCH_MEM_PER_NODE"
echo "  --array=0-$N_LAST_ARRAYID"
echo "  --nodes=$S_BATCH_NODES"
echo "  --ntasks-per-node=$S_BATCH_N_TASKS_PER_NODE"
echo "  --gres=gpu:$S_BATCH_GPUS"
echo "  $PATH_PARENT/slurm_job_array/$NAME_JOB_ARRAY_SCRIPT"
echo


sbatch \
  --job-name="$NAME_FOLDER_CONFIGS" \
  --array=0-"$N_LAST_ARRAYID" \
  --output="$PATH_LOG_DIR"/job_array_launcher_%A_%a.out \
  --error="$PATH_LOG_DIR"/job_array_launcher_%A_%a.err \
  --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",PATH_FOLDER_CONFIGS="$PATH_FOLDER_CONFIGS",WORKDIR="$WORKDIR",MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI",PATH_CONTAINER="$PATH_CONTAINER",PATH_CONTENT_ROOT="$PATH_CONTENT_ROOT",PATH_CONTAINER_CONTENT_ROOT="$PATH_CONTAINER_CONTENT_ROOT",PATH_CACHE_DIR_HPC="$PATH_CACHE_DIR_HPC" \
  --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
  --mem="$S_BATCH_MEM_PER_NODE" \
  --gres=gpu:"$S_BATCH_GPUS" \
  --time="$S_BATCH_TIME" \
  --partition="$S_BATCH_PARTITION" \
  "$PATH_PARENT"/slurm_job_array/"$NAME_JOB_ARRAY_SCRIPT"

# Note: The following options are not needed with Ruche
#  --qos="$S_BATCH_QOS" \
#  --account="$S_BATCH_ACCOUNT" \
