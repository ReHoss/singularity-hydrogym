#!/bin/bash

# Name of the project
NAME_PROJECT="singularity-hydrogym"
# Name of the job script
NAME_JOB_SCRIPT="run_python_script_singularity.slurm"

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

PATH_PYTHON_SCRIPT="$PATH_CONTAINER_CONTENT_ROOT"/examples/generate_natural_states/generate_natural_states.py

# Firedrake writes to the cache directory that is replicated on the Singularity container
# this triggers an OSError: [Errno 28] No space left on device
PATH_HOME_DIR_HPC="/gpfs/users/hosseinkhanr"
PATH_CACHE_DIR_HPC="$PATH_HOME_DIR_HPC"/.cache



# Get the venv name from the command line
V_ENV_NAME="venv_control_dde"
# Get the path of the virtual environment
PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
echo PATH_VENV_BIN: "$PATH_VENV_BIN"
echo
# Activate the virtual environment, if working echo the name of the venv
# shellcheck source=/home/hosseinkhan/Documents/work/phd/git_repositories/doe4rl/venv/venv_control_dde/bin/activate
source "$PATH_VENV_BIN" && echo "Activation of virtual environment: $V_ENV_NAME"
echo


# Get the folder name from the command line, and the arguments to pass to the python script
while getopts 'p:a:' flag; do
  case "${flag}" in
  p) PATH_PYTHON_SCRIPT="${OPTARG}" ;;
  a) ARGS_PYTHON_SCRIPT="${OPTARG}" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done

# Check PATH_PYTHON_SCRIPT is not empty
if [ -z "$PATH_PYTHON_SCRIPT" ]; then
  echo Missing option.s.
  exit
fi

# Get the basename of the python script without the extension
BASENAME_SCRIPT=$(basename "$PATH_PYTHON_SCRIPT" .py)
echo "Script basename: $BASENAME_SCRIPT"
echo

# Create the name of the log directory with the current date and time
PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$BASENAME_SCRIPT"/$(date +"%Y-%m-%d_%H-%M-%S")

echo "Log directory: $PATH_LOG_DIR"
echo

# Create the log directory for the current config file
mkdir -p "$PATH_LOG_DIR"/"$CONFIG_FILE_NAME"

# Launch the job array script
echo "Launching $NAME_JOB_SCRIPT"
echo

echo PATH_PYTHON_SCRIPT: "$PATH_PYTHON_SCRIPT"
echo
echo ARGS_PYTHON_SCRIPT: "$ARGS_PYTHON_SCRIPT"
echo

# Set defaults values for the sbatch options
# --- Number of CPUs per task ---
S_BATCH_CPU_PER_TASK=1

# --- Time limit ---
S_BATCH_TIME=19:59:00
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
S_BATCH_MEM_PER_NODE="16G"

echo "sbatch options:"
echo "  --job-name=$BASENAME_SCRIPT"
echo "  --output=$PATH_LOG_DIR/%j.out"
echo "  --error=$PATH_LOG_DIR/%j.err"
echo "  --export=NAME_PROJECT=$NAME_PROJECT,PATH_PYTHON_SCRIPT=$PATH_PYTHON_SCRIPT,ARGS_PYTHON_SCRIPT=$ARGS_PYTHON_SCRIPT"
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
#echo "  --ntasks=$S_BATCH_SLURM_NTASKS"
echo "  --time=$S_BATCH_TIME"
echo "  --partition=$S_BATCH_PARTITION"
echo "  --account=$S_BATCH_ACCOUNT"
echo "  --mem=$S_BATCH_MEM_PER_NODE"
echo "  --nodes=$S_BATCH_NODES"
echo "  --ntasks-per-node=$S_BATCH_N_TASKS_PER_NODE"
echo "  --gres=gpu:$S_BATCH_GPUS"
echo "  $PATH_PARENT/slurm_script/$NAME_JOB_SCRIPT"
echo


sbatch \
  --job-name="$BASENAME_SCRIPT" \
  --output="$PATH_LOG_DIR"/%j.out \
  --error="$PATH_LOG_DIR"/%j.err \
  --export=PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",ARGS_PYTHON_SCRIPT="$ARGS_PYTHON_SCRIPT",PATH_CONTAINER="$PATH_CONTAINER",PATH_CONTENT_ROOT="$PATH_CONTENT_ROOT",PATH_CONTAINER_CONTENT_ROOT="$PATH_CONTAINER_CONTENT_ROOT",PATH_CACHE_DIR_HPC="$PATH_CACHE_DIR_HPC" \
  --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
  --mem="$S_BATCH_MEM_PER_NODE" \
  --time="$S_BATCH_TIME" \
  --partition="$S_BATCH_PARTITION" \
  "$PATH_PARENT"/slurm_script/"$NAME_JOB_SCRIPT"



# Note: The following options are not needed with Ruche
# it is not easy to get the slurm account name plainly

#  --qos="$S_BATCH_QOS" \
#  --account="$S_BATCH_ACCOUNT" \
