PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT="$PATH_PARENT/../.."
PATH_SINGULARITY_DEFINITION_FILE_DIR="$PATH_CONTENT_ROOT"/singularity

NAME_SINGULARITY_DEFINITION_FILE="firedrake-hydrogym.def"

NAME_IMAGE_SIF_FILE="hydrogym-firedrake.sif"

singularity build \
--no-cleanup \
--fakeroot \
"$NAME_IMAGE_SIF_FILE" \
"$PATH_SINGULARITY_DEFINITION_FILE_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE"
