import argparse
import tempfile

import yaml
import pathlib
import mlflow
import hydrogym.core
import numpy as np
import firedrake  # pyright: ignore [reportMissingImports]

from typing import Any, Optional
from singularity_hydrogym.environments import navierstokes2d
from singularity_hydrogym.utils import utils


def log_config_artifact_no_mlflow(path_output_data: str, dict_config: dict[str, Any]):
    """
    Log the configuration file as an artifact without using MLFlow.
    """
    path_config = pathlib.Path(
        f"{path_output_data}/artifact_nomlflow_config/config.yaml"
    )
    path_config.parent.mkdir(parents=True, exist_ok=False)
    with open(path_config, "w") as file:
        yaml.dump(dict_config, file)
        path_config_absolute = pathlib.Path(file.name)
        return path_config_absolute


def flatten_dict(nested_dict, parent_key="", sep="."):
    """Utility function to flatten nested dictionaries."""
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compute_steps_from_seconds(duration, dt) -> int:
    return int(np.ceil(duration / dt))


def write_flow(
    flow: hydrogym.firedrake.FlowConfig,
    reynolds_number: float | int,
    path_output_data: Optional[str] = None,
) -> None:
    if isinstance(reynolds_number, float):
        reynolds_number = int(reynolds_number)

    # Create output directory with Pathlib
    if path_output_data is not None:
        pathlib.Path(path_output_data).mkdir(parents=True, exist_ok=True)

    pressure = flow.p
    velocity = flow.u
    vorticity = flow.vorticity()

    with tempfile.TemporaryDirectory() as tmpfile:
        if path_output_data is None:
            path_output_data = tmpfile
        # Save the solution
        flow.save_checkpoint(f"{path_output_data}/{reynolds_number}_natural.h5")
        # noinspection PyUnresolvedReferences
        pvd = firedrake.output.VTKFile(
            f"{path_output_data}/{reynolds_number}_natural.pvd"
        )
        pvd.write(velocity, pressure, vorticity)


def step_environment(hydrogym_env: hydrogym.core.FlowEnv, n_steps: int) -> None:
    assert (float(n_steps)).is_integer(), "n_steps must be an integer"
    n_steps = int(n_steps)
    # Derive a null action from the action space
    array_action = hydrogym_env.action_space.sample() * 0.0
    for i in range(n_steps):
        hydrogym_env.step(array_action)


def run_simulation_natural_state(
    dict_config: dict[str, Any], path_mlflow_uri: pathlib.Path
):
    """Runs a simulation and logs parameters and outputs to MLFlow."""
    # Load the configuration as a dictionary
    dict_params = dict_config

    # Get the name of the experiment
    name_xp_mlflow: str = dict_params["mlflow_experiment_name"]
    print(f"Name of the experiment: {name_xp_mlflow}")
    print(f"Setting up MLFlow tracking uri: {path_mlflow_uri}")
    # Set the path where data will be stored
    mlflow.set_tracking_uri(f"file:{path_mlflow_uri}")
    print(f"Setting up MLFlow experiment: {name_xp_mlflow}")
    # Set the name of the experiment
    mlflow.set_experiment(name_xp_mlflow)
    ml_flow_experiment = mlflow.get_experiment_by_name(name_xp_mlflow)
    # Run the main function with the ID corresponding to the experiment
    ml_flow_experiment_id: str = ml_flow_experiment.experiment_id  # pyright: ignore
    # [reportOptionalMemberAccess]
    with mlflow.start_run(experiment_id=ml_flow_experiment_id) as run:
        # MlFlow logging, formatting names of dict_params
        print(f"Starting run {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")

        # Create the path and directory where to store data if needed
        path_generated_data = (
            f"{path_mlflow_uri}/{run.info.experiment_id}/"
            f"{run.info.run_id}/generated_data"
        )
        pathlib.Path(path_generated_data).mkdir()

        # Flatten the config dictionary
        dict_config_flattened = flatten_dict(nested_dict=dict_config)
        # Store each of the config parameters in the mlflow run
        for key, value in dict_config_flattened.items():
            mlflow.log_param(key, value)

        # MLFlow logging: each run has a unique ID
        mlflow.log_param("_id", run.info.run_id)
        path_config = pathlib.Path(f"{path_generated_data}/config/config.yaml")
        path_config.parent.mkdir(parents=True, exist_ok=False)
        with open(path_config, "w") as file:
            yaml.dump(dict_config, file)
            log_config_artifact_no_mlflow(
                path_output_data=path_generated_data, dict_config=dict_config
            )

        dict_env = dict_config["env"]
        dict_pde_config = dict_env["parameters"]["dict_pde_config"]
        name_flow = dict_env["name"]
        reynolds_number = dict_pde_config["reynolds"]
        name_mesh_resolution = dict_pde_config["mesh"]
        stabilization = dict_env["parameters"]["dict_solver"]["stabilization"]
        duration_seconds = dict_config["duration_seconds"]
        dt = dict_pde_config["dt"]

        print(
            f"Computing natural state for {name_flow} at Re={reynolds_number} "
            f"with mesh resolution {name_mesh_resolution} and stabilization {stabilization}"
        )

        # Output directory
        name_directory = "_".join(
            [name_flow, str(reynolds_number), name_mesh_resolution, stabilization]
        )
        path_output_data = (
            f"{utils.PATH_PROJECT_ROOT}/data/natural_state"
            f"/{name_flow}/{name_directory}"
        )
        # Create output directory with Pathlib
        pathlib.Path(path_output_data).mkdir(parents=True, exist_ok=True)

        # Create hydrogym config
        dict_hydrogym_config = utils.create_navierstokes2d_dict_config(
            dict_yaml_config=dict_config,
            path_output_data=path_output_data,
        )

        # Create hydrogym environment
        gym_env = navierstokes2d.NavierStokesFlow2D(**dict_hydrogym_config)

        # Step for the given duration
        n_steps: int = compute_steps_from_seconds(duration_seconds, dt)
        step_environment(gym_env, n_steps)

        # Write the flow field to the right directory
        # noinspection PyTypeChecker
        write_flow(
            gym_env.flow,  # pyright: ignore [reportArgumentType]
            reynolds_number,
            path_output_data,
        )


def main():
    # Parsing arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "-y",
        "--yaml",
        type=str,
        required=True,
        help="Absolute path to the YAML configuration file.",
    )
    dict_args: argparse.Namespace = parser.parse_args()
    path_config = dict_args.yaml
    assert path_config is not None, "No .yaml config file provided."
    path_config: pathlib.Path = pathlib.Path(path_config)
    # Get the path of the current `script`
    path_current_script: pathlib.Path = pathlib.Path(__file__).parent
    # Get the path of the project root
    path_project_root: pathlib.Path = path_current_script.parent.parent.resolve()
    # Log the path of the project root
    print(f"Path of the project root: {path_project_root}")
    path_config_absolute: pathlib.Path = path_project_root / path_config

    with open(path_config_absolute, "r") as file:
        dict_config: dict[str, Any] = yaml.safe_load(file)

    # Get the path where the mlflow data will be stored
    path_mlflow_uri: pathlib.Path = pathlib.Path(
        path_project_root / "data" / "mlruns"
    ).resolve()

    # Run the simulation
    run_simulation_natural_state(
        dict_config=dict_config, path_mlflow_uri=path_mlflow_uri
    )


if __name__ == "__main__":
    main()
