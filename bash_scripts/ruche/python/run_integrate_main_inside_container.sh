PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")

# Augment python path with the content root
export PYTHONPATH="$PATH_CONTENT_ROOT"
