PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")

PATH_VENV_FIREDRAKE="/home/firedrake/firedrake/bin/activate"

# Augment python path with the content root
export PYTHONPATH="$PATH_CONTENT_ROOT"

# Activate the firedrake virtual environment
source "$PATH_VENV_FIREDRAKE"

# Run the main script
python3 "$PATH_CONTENT_ROOT/singularity_hydrogym/integration/main.py" --yaml "$PATH_CONTENT_ROOT/configs/cavity/cavity_reynolds-7500.yaml"