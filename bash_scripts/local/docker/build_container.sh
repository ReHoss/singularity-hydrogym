# Description: Build a container with the current user's UID and GID

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")
PATH_DOCKERFILE_DIR="$PATH_CONTENT_ROOT"/docker
TAG_IMAGE="hydrogym-firedrake"


echo "Building the Docker image from the Dockerfile in $PATH_DOCKERFILE_DIR"
echo "The image will be tagged as $TAG_IMAGE"
echo "The current user's UID and GID will be passed to the Dockerfile"
echo

echo "Command:
  docker build \
  --tag $TAG_IMAGE \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --progress=plain \
  $PATH_DOCKERFILE_DIR"

# Change the current directory to the project root in order to give the
# correct context to the Docker build command
cd "$PATH_CONTENT_ROOT" || exit

docker build \
  --tag "$TAG_IMAGE" \
  --build-arg USER_ID="$(id -u)" \
  --build-arg GROUP_ID="$(id -g)" \
  --progress=plain \
  . \
  --file "$PATH_DOCKERFILE_DIR/Dockerfile"



# --tag hydrogym-firedrake: Name of the image
# --build-arg USER_ID=$(id -u): Pass the current user's UID to the Dockerfile
# --build-arg GROUP_ID=$(id -g): Pass the current user's GID to the Dockerfile
# --progress=plain: Show the build progress in a plain format (add verbosity)
# .  : Path to the context of the build (the project root) to give the Dockerfile access to the project files
# such that directives like COPY can work
# --file "$PATH_DOCKERFILE_DIR": Path to the directory containing the Dockerfile to build the image from