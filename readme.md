
Folders description:

- `bash_scripts/` - Contains scripts for running containers and others. 
- `congigs/` - Contains configuration files for the `src/` python scripts.
- `data/` - Contains data needed by the `src/` python scripts.
- `docker/` - Contains the Dockerfiles for the containers.
- `singularity/` - Contains the Singularity files for the containers.
- `src/` - The directory `venv` may content your own virtual environment, to run MLFlow and other tools or a version of the Firedrake environment.


Data generated from `src/integration/main.py` will be saved in the `data/mlruns/` folder.
The `data/` folder is mounted in the containers (in read-write mode), so the data will be available in the containers.

##### Docker
Usually, Docker is not supported on HPC clusters.
However, an equivalent solution is to use Singularity.
The Docker images available in this repository are analogous to the Singularity images available in the `singularity/` folder.
It can be used for local development and testing through development environments such as VSCode or PyCharm.

##### Singularity
The Singularity containers are the ones that should be used on HPC clusters.
To build a Singularity container, scripts from `bash_scripts/local/singularity/` can be used.

###### Details on the images
The Docker images are based on Firedrake official images.
Those images create a user called _firedrake_.
The Firedrake python environment is located at `/home/firedrake/firedrake/bin/activate`.
In order to let users in the container to activate this environment and manipulate files in `/home/firedrake`, the _firedrake_ user is modified to have the same UID as the user running the containers during image generation.
Such images are defined in `docker/Dockerfile` and `singularity/definition_files/hydrogym-firedrake.def`.
However, user account modifications through `usermod` may be not allowed in some HPC clusters, thus the images may not work for security reasons.

Consequently, an alternative solution implemented in `singularity/definition_files/hydrogym-firedrake_nousernamespace.def` is to make the `\home\firedrake` directory writable and executable by all users.
It is the preferable way to go.

### Installation
Depending on the user needs, the following software are required:
- Docker
- Singularity
- MLFlow


TODO:
- Issue update the repo for Firedrake warnings
- PATH_VENV="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
