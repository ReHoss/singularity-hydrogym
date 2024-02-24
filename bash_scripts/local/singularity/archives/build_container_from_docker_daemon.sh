#PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
#PATH_CONTENT_ROOT="$PATH_PARENT/../../.."
#PATH_SINGULARITY_DEFINITION_FILE_DIR="$PATH_CONTENT_ROOT"/singularity
#
#NAME_SINGULARITY_DEFINITION_FILE="firedrake-hydrogym.def"
#
#singularity build \
#--no-cleanup \
#"$TAG_IMAGE".sif \
#docker-daemon://"$TAG_IMAGE":latest