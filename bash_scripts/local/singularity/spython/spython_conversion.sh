# Translate a Dockerfile to a Singularity .sif image file
# TODO: Add verbosity

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../../..")

PATH_SPYTHON_VENV_ACTIVATION_SCRIPT="$PATH_PARENT"/.venv/bin/activate


PATH_DOCKERFILE_DIR="$PATH_CONTENT_ROOT"/docker
PATH_SINGULARITY_SIF_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/definition_files

NAME_DOCKERFILE="Dockerfile"
NAME_SINGULARITY_DEFINITION_FILE="hydrogym-firedrake.def"

# Load venv
source "$PATH_SPYTHON_VENV_ACTIVATION_SCRIPT"


if [ ! -f "$PATH_SINGULARITY_SIF_FILE_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE" ]; then
  echo "Creating the Singularity definition file $NAME_SINGULARITY_DEFINITION_FILE"
  touch "$PATH_SINGULARITY_SIF_FILE_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE"
fi

spython recipe \
"$PATH_DOCKERFILE_DIR"/"$NAME_DOCKERFILE" > "$PATH_SINGULARITY_SIF_FILE_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE"

deactivate