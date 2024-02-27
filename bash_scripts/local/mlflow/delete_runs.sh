#!/bin/bash

# Path of the parent directory of this script
PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
# Path of the root of the project
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")

V_ENV_NAME="venv_monitoring_tools"
PATH_VENV="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
# Path of the "store", i.e., where the mlflow runs are stored
PATH_BACKEND_STORE="$PATH_CONTENT_ROOT"/data/mlruns

# Load virtual environment
source "$PATH_VENV"

echo "Chosen backend store: $PATH_BACKEND_STORE"
# Run mlflow
mlflow gc --backend-store-uri "$PATH_BACKEND_STORE"
