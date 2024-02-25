# In Singularity 3, environment variables cannot be passed to the definition file .def dynamically.
# Therefore, the user and group IDs must be hardcoded in the definition file.
# The user and group IDs are passed in order to give the same permissions to the user in the container as on the host.

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")

NAME_SINGULARITY_DEFINITION_FILE="hydrogym-firedrake_nousernamespace.def"
PATH_SINGULARITY_DEFINITION_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/definition_files
PATH_SINGULARITY_DEFINITION_FILE="$PATH_SINGULARITY_DEFINITION_FILE_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE"

# User and group IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)

NAME_IMAGE_SIF_FILE="hydrogym-firedrake_nousernamespace_uid-${USER_ID}_gid-${GROUP_ID}_hostname-$(hostname).sif"
PATH_SIF_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/images
PATH_SIF_FILE="$PATH_SIF_FILE_DIR"/"$NAME_IMAGE_SIF_FILE"

# Build the Singularity image
singularity build \
--no-cleanup \
--fakeroot \
"$(readlink -f "$PATH_SIF_FILE")" \
"$(readlink -f "$PATH_SINGULARITY_DEFINITION_FILE")"

# If the getopts command is used, delete the temporary directory
if [ -n "$opt" ]; then
  rm -r "$TEMP_DIR"
fi

# https://github.com/apptainer/singularity/issues/5941
