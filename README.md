## Installation

[Hydrogym](https://github.com/dynamicslab/hydrogym) uses a Computational Fluid Dynamics (CFD) library called[Firedrake](https://www.firedrakeproject.org/download.html).
Firedrake is a Python library that solves partial differential equations using the finite element method. It is not easy to install on platforms where the user does not have root access, such as HPC clusters.

Depending on the user needs, the following softwares may be  required:
- _Docker_
- _Singularity_
- _MLFlow_: Experiment tracker that is installed by default from `pyproject.toml` file.
- _Stable Baselines3_: Reinforcement learning library that is also installed by default.
- [_spython_](https://singularityhub.github.io/singularity-cli/): to convert Dockerfile to Singularity definition files `.def`

### Local development

Locally, you can install this codebase by following these steps:

- Install Firedrake: https://www.firedrakeproject.org/download.html
- The firedrake installation necessarily creates a virtual environment (_e.g._ `venv_firedrake`).
- Activate the virtual environment: `source venv_firedrake/bin/activate`
- Install the `singularity-hydrogym` package: `pip install -e .`

### Docker

For local development, Docker can be used.

To build the Docker image, run the following script (from the project root):

`./singularity-hydrogym/bash_scripts/local/docker/build_container.sh`

This scripts builds the Docker image from the Dockerfile located in `docker/`. It passes the `--build-arg` flag to the Docker build command to specify the UID of the host user running the script. This permits to give the same UID to the _firedrake_ user in the container, so that writing permissions from the container to the host are granted.

To run the container run the following script (from the project root):

`./singularity-hydrogym/bash_scripts/local/docker/run_container.sh`

Now, inside the container you can perform the integration with this following example script:

`python src/integration/main.py --yaml /home/firedrake/mount_dir/project_root/configs/cavity/cavity_reynolds-7500.yaml`

Note that while `src/` is copied to the container file-system during the **build** phase of the Docker image, the `data/` and `configs/` directories are mounted in the container when running it. 

Usually, Docker is not supported on HPC clusters.
However, an equivalent solution is to use Singularity.
The Docker images available in this repository are analogous to the Singularity images available in the `singularity/` folder.
It can be used for local development and testing through development environments such as VSCode or PyCharm.

### Singularity
The Singularity containers are the ones that should be used on HPC clusters.
To build a Singularity container, scripts from `bash_scripts/local/singularity/` can be used.





##### Details on the images
The Docker images are based on Firedrake official images.
Those images create a user called _firedrake_.
The Firedrake python environment is located at `/home/firedrake/firedrake/bin/activate`.
In order to let users in the container to activate this environment and manipulate files in `/home/firedrake`, the _firedrake_ user is modified to have the same UID as the user running the containers during image generation.
Such images are defined in `docker/Dockerfile` and `singularity/definition_files/hydrogym-firedrake.def`.
However, user account modifications through `usermod` may be not allowed in some HPC clusters, thus the images may not work for security reasons.

Consequently, an alternative solution implemented in `singularity/definition_files/hydrogym-firedrake_nousernamespace.def` is to make the `\home\firedrake` directory writable and executable by all users.
It is the preferable way to go.




## Codebase description:

- `bash_scripts/` - Contains scripts for running containers and others. 
- `congigs/` - Contains configuration files for the `src/` python scripts.
- `data/` - Contains data needed by the `src/` python scripts.
- `docker/` - Contains the Dockerfiles for the containers.
- `singularity/` - Contains the Singularity files for the containers.
- `src/` - The directory `venv` may content your own virtual environment, to run MLFlow and other tools or a version of the Firedrake environment.


Data generated from `src/integration/main.py` will be saved in the `data/mlruns/` folder.
The `data/` folder is mounted in the containers (in read-write mode), so the data will be available in the containers.

### Initial vector field

Initial vector fields are needed to start the simulations.
Only the one for the Cavity Flow problem is provided in the `data/` folder for now but more can be added very easily.







TODO:
- Issue update the repo for Firedrake warnings
- PATH_VENV="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
- Talk about initial data
- Make the Steady State solver script
- Containers run on Ruche but not on local now ?
- Sync with cluster
- Add `pip install -e .` in the Dockerfile

Ressources:
- https://docs.archer2.ac.uk/user-guide/containers/#running-parallel-mpi-jobs-using-singularity-containers
- https://nemo-related.readthedocs.io/en/latest/compilation_notes/singularity_firedrake.html
- https://docs.sylabs.io/guides/latest/user-guide/definition_files.html
- Slack (Singularity, Firedrake?)


