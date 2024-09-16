#!/bin/bash

module load singularity/3.8.3/gcc-11.2.0

#SBATCH --job-name=batch_experiment_hosseinkhan_default # Name of the job by default

#SBATCH --mail-type=ALL # The points during the execution to send an email
#SBATCH --mail-user=remy.hosseinkhan@lisn.fr


# Check if environment variables are defined
if [ -z "$PATH_CONTAINER" ]; then
    echo "The environment variable PATH_CONTAINER is not defined"
    exit 1
fi

if [ -z "$PATH_CONTENT_ROOT" ]; then
    echo "The environment variable PATH_CONTENT_ROOT is not defined"
    exit 1
fi

if [ -z "$PATH_CACHE_DIR_HPC" ]; then
    echo "The environment variable PATH_CACHE_DIR_HPC is not defined"
    exit 1
fi

if [ -z "$PATH_CONTAINER_CONTENT_ROOT" ]; then
    echo "The environment variable PATH_CONTAINER_CONTENT_ROOT is not defined"
    exit 1
fi


#if [ -z "$NAME_PROJECT" ]; then
#    echo "The environment variable NAME_PROJECT is not defined"  # TODO: Change
#    exit 1
#fi

if [ -z "$PATH_PYTHON_SCRIPT" ]; then
    echo "The environment variable PATH_PYTHON_SCRIPT is not defined"
    exit 1
fi

#if [ -z "$WORKDIR" ]; then
#    echo "The environment variable WORKDIR is not defined"
#    exit 1
#fi

PATH_PARENT=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)

# Print some path variables
echo Starting "$SLURM_JOB_NAME" on "$SLURMD_NODENAME" with the following parameters:
echo PWD: "$PWD"
echo WORKDIR: "$WORKDIR"
echo PATH_PARENT: "$PATH_PARENT"
echo
# Echo the number of processors and more information
echo Number of processors: "$NPROCS"
echo
echo  SLURM_SUBMIT_HOST: "$SLURM_SUBMIT_HOST"
echo  SLURM_JOB_QOS: "$SLURM_JOB_QOS"
echo  SLURM_SUBMIT_DIR "$SLURM_SUBMIT_DIR"
echo  SLURM_JOB_ID "$SLURM_JOB_ID"
echo  SLURM_JOB_NAME "$SLURM_JOB_NAME"
echo  SLURM_JOB_ACCOUNT "$SLURM_JOB_ACCOUNT"
echo  SLURM_JOB_PARTITION "$SLURM_JOB_PARTITION"
echo  SLURM_JOB_NODELIST "$SLURM_JOB_NODELIST"
echo  SLURM_JOB_NUM_NODES "$SLURM_JOB_NUM_NODES"
echo  SLURM_NTASKS "$SLURM_NTASKS"
echo  SLURM_NTASKS_PER_NODE "$SLURM_NTASKS_PER_NODE"
echo  SLURM_NTASKS_PER_SOCKET "$SLURM_NTASKS_PER_SOCKET"
echo  SLURM_NTASKS_PER_CORE "$SLURM_NTASKS_PER_CORE"
echo  SLURM_MEM_PER_CPU "$SLURM_MEM_PER_CPU"
echo  SLURM_MEM_PER_NODE "$SLURM_MEM_PER_NODE"
echo  SLURM_CPUS_ON_NODE "$SLURM_CPUS_ON_NODE"
echo  SLURM_JOB_CPUS_PER_NODE "$SLURM_JOB_CPUS_PER_NODE"
echo  SLURM_RESTART_COUNT "$SLURM_RESTART_COUNT"
echo  SLURM_CHECKPOINT_IMAGE_DIR "$SLURM_CHECKPOINT_IMAGE_DIR"
echo


# Split the arguments of the python script with mapfile
# ARGS_PYTHON_SCRIPT environment variable is also broadcasted
IFS=" " read -r -a LIST_ARGS_PYTHON_SCRIPT <<< "$ARGS_PYTHON_SCRIPT"

# Run the main script
echo "Container execution: $PATH_CONTAINER"
echo
echo Running the script "$PATH_PYTHON_SCRIPT" with the following arguments: "$ARGS_PYTHON_SCRIPT"
echo
echo "Binding the following directories:"
echo "  - $PATH_CONTENT_ROOT -> $PATH_CONTAINER_CONTENT_ROOT"
echo "  - $PATH_CACHE_DIR_HPC -> $PATH_CACHE_DIR_HPC"
echo

singularity exec \
  --no-home \
  --writable-tmpfs \
  --no-init \
  --bind "$PATH_CONTENT_ROOT":"$PATH_CONTAINER_CONTENT_ROOT" \
  --bind "$PATH_CACHE_DIR_HPC":"$PATH_CACHE_DIR_HPC" \
  "$PATH_CONTAINER" /bin/bash -c ". /home/firedrake/firedrake/bin/activate && cd $PATH_CONTAINER_CONTENT_ROOT && python $PATH_PYTHON_SCRIPT" "${LIST_ARGS_PYTHON_SCRIPT[@]}"

echo
#singularity exec <container.sif> /bin/bash -c "source /path/to/venv/bin/activate && python your_script.py"
