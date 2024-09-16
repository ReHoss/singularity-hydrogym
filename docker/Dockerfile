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
USER firedrake
# Source the firedrake environment
ENV PATH_FIREDRAKE_VENV_ACTIVATION_SCRIPT=/home/firedrake/firedrake/bin/activate

# Create a directory for mounting
RUN mkdir -p /home/firedrake/mount_dir/project_root/

RUN /bin/bash -c "source $PATH_FIREDRAKE_VENV_ACTIVATION_SCRIPT && \
    pip install --upgrade pip==24.0 && \
    pip install gymnasium \
                git+https://github.com/ReHoss/hydrogym.git@feature/add_back_coarse_meshes \
                jsonschema \
                matplotlib \
                mlflow==1.27.0 \
                numpy \
                pandas \
                pyyaml \
                sb3-contrib \
                scipy \
                tensorboard \
                tqdm \
    "



WORKDIR /home/firedrake/mount_dir/project_root


# Set the PYTHONPATH
ENV PYTHONPATH="/home/firedrake/mount_dir/project_root:$PYTHONPATH"

# PRINT THE CURRENT ENVIRONMENT TO THE TERMINAL
RUN /bin/bash -c "echo 'Current python: $(which python)' && \
    echo 'Current pip: $(which pip)' && \
    echo 'Current python version: $(python --version)' && \
    echo 'Current pip version: $(pip --version)' && \
    echo 'Current pip list: $(pip list)'"


ENTRYPOINT ["/bin/bash", "-l", "-c", "source /home/firedrake/firedrake/bin/activate && /bin/bash"]
