# In Singularity 3, environment variables cannot be passed to the definition file .def dynamically.
# Therefore, the user and group IDs must be hardcoded in the definition file.
# The user and group IDs are passed in order to give the same permissions to the user in the container as on the host.

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../..")

NAME_SINGULARITY_DEFINITION_FILE="hydrogym-firedrake.def"
PATH_SINGULARITY_DEFINITION_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/definition_files
PATH_SINGULARITY_DEFINITION_FILE="$PATH_SINGULARITY_DEFINITION_FILE_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE"

# User and group IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Overwrite user and group IDs if provided
while getopts ":u:g:" opt; do
  case $opt in
    u) USER_ID=$OPTARG ;;
    g) GROUP_ID=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# If the getopts command is used, create a temporary directory to store the Singularity definition file
# and the Singularity image file
# This is necessary because the Singularity build command does not accept the --output option
# and the Singularity definition file and the Singularity image file must be in the same directory
# The temporary directory will be deleted after the Singularity build command
if [ -n "$opt" ]; then
  TEMP_DIR=$(mktemp -d)
  PATH_SINGULARITY_DEFINITION_FILE_TMP="$TEMP_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE"
  cp "$PATH_SINGULARITY_DEFINITION_FILE" "$PATH_SINGULARITY_DEFINITION_FILE_TMP"
  # Replace the user and group IDs in the Singularity definition file
  sed -i "s/{{ USER_ID }}/$USER_ID/g" "$PATH_SINGULARITY_DEFINITION_FILE_TMP"
  sed -i "s/{{ GROUP_ID }}/$GROUP_ID/g" "$PATH_SINGULARITY_DEFINITION_FILE_TMP"
else
  PATH_SINGULARITY_DEFINITION_FILE_TMP="$PATH_SINGULARITY_DEFINITION_FILE"
fi

NAME_IMAGE_SIF_FILE="hydrogym-firedrake-uid-$USER_ID-gid-$GROUP_ID.sif"
PATH_SIF_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/images
PATH_SIF_FILE="$PATH_SIF_FILE_DIR"/"$NAME_IMAGE_SIF_FILE"

# Build the Singularity image
singularity build \
--no-cleanup \
--fakeroot \
"$(readlink -f "$PATH_SIF_FILE")" \
"$(readlink -f "$PATH_SINGULARITY_DEFINITION_FILE_TMP")"

# If the getopts command is used, delete the temporary directory
if [ -n "$opt" ]; then
  rm -r "$TEMP_DIR"
fi
