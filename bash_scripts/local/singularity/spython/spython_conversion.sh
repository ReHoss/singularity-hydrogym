# Translate a Dockerfile to a Singularity .sif image file
# TODO: Add verbosity

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")

PATH_SPYTHON_VENV_ACTIVATION_SCRIPT="$PATH_PARENT"/.venv/bin/activate


PATH_DOCKERFILE_DIR="$PATH_CONTENT_ROOT"/docker
PATH_SINGULARITY_SIFE_FILE_DIR="$PATH_CONTENT_ROOT"/singularity

NAME_DOCKERFILE="Dockerfile"
NAME_SINGULARITY_DEFINITION_FILE="firedrake-hydrogym.def"

# Load venv
source "$PATH_SPYTHON_VENV_ACTIVATION_SCRIPT"

spython recipe \
"$PATH_DOCKERFILE_DIR"/"$NAME_DOCKERFILE" > "$PATH_SINGULARITY_SIFE_FILE_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE"

deactivate