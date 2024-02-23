PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT="$PATH_PARENT/../.."

NAME_SINGULARITY_DEFINITION_FILE="hydrogym-firedrake.def"
PATH_SINGULARITY_DEFINITION_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/definition_files
PATH_SINGULARITY_DEFINITION_FILE="$PATH_SINGULARITY_DEFINITION_FILE_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE"

# User and group IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)

NAME_IMAGE_SIF_FILE="hydrogym-firedrake-uid-$USER_ID-gid-$GROUP_ID.sif"
PATH_SIF_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/images
PATH_SIF_FILE="$PATH_SIF_FILE_DIR"/"$NAME_IMAGE_SIF_FILE"

singularity build \
--no-cleanup \
--fakeroot \
--build-arg USER_ID="$USER_ID" \
--build-arg GROUP_ID="$GROUP_ID" \
"$(readlink -f "$PATH_SIF_FILE")" \
"$(readlink -f "$PATH_SINGULARITY_DEFINITION_FILE")"
