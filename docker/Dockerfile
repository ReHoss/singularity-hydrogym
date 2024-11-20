FROM firedrakeproject/firedrake-vanilla:2024-04
LABEL authors="hosseinkhan"

#====================
# Firedrake change uid approach
USER root
ARG USER_ID
ARG GROUP_ID
# TODO: Maybe set this at the top of the file is enough, and faster?
RUN usermod -u $USER_ID firedrake
RUN groupmod -g $GROUP_ID firedrake

#====================
# Switch to the firedrake user
USER firedrake
# Source the firedrake environment
ENV PATH_FIREDRAKE_VENV_ACTIVATION_SCRIPT=/home/firedrake/firedrake/bin/activate

# Install git-lfs to handle large files in HydroGym
USER root
RUN apt-get update && apt-get install -y git-lfs && git lfs install && apt-get clean

# Switch back to the firedrake user
USER firedrake
#@feature/add_back_coarse_meshes
# Clone and install HydroGym with mesh files using git-lfs
RUN /bin/bash -c "source $PATH_FIREDRAKE_VENV_ACTIVATION_SCRIPT && \
    pip install --upgrade pip==24.0 && \
    git clone https://github.com/ReHoss/hydrogym.git /home/firedrake/hydrogym && \
    cd /home/firedrake/hydrogym && \
    git checkout origin/feature/add_back_coarse_meshes && \
    git lfs install && git lfs fetch --all && \
    pip install ."

# Install additional Python packages
RUN /bin/bash -c "source $PATH_FIREDRAKE_VENV_ACTIVATION_SCRIPT && \
    pip install gymnasium \
                jsonschema \
                matplotlib \
                mlflow==1.27.0 \
                numpy \
                pandas \
                pyyaml \
                sb3-contrib \
                scipy \
                tensorboard \
                tqdm"

# Create a directory for mounting
RUN mkdir -p /home/firedrake/mount_dir/project_root/

WORKDIR /home/firedrake/mount_dir/project_root

ENV PYTHONPATH="/home/firedrake/mount_dir/project_root:$PYTHONPATH"

# PRINT THE CURRENT ENVIRONMENT TO THE TERMINAL
RUN /bin/bash -c "echo 'Current python: $(which python)' && \
    echo 'Current pip: $(which pip)' && \
    echo 'Current python version: $(python --version)' && \
    echo 'Current pip version: $(pip --version)' && \
    echo 'Current pip list: $(pip list)'"

# ENTRYPOINT to activate the Firedrake environment
ENTRYPOINT ["/bin/bash", "-l", "-c", "source /home/firedrake/firedrake/bin/activate && /bin/bash"]
