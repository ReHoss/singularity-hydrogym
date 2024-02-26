import argparse
import pathlib
import random
import numpy as np
import mlflow
import pandas as pd
import yaml

import hydrogym.firedrake as hgym

from src.utils import utils


def create_hydrogym_dict_config(name_flow: str,
                                dict_parameters: dict,
                                path_output_data: str):
    assert name_flow in utils.LIST_STR_FLOWS, "Invalid flow name"
    assert "dict_pde_config" in dict_parameters.keys()
    assert "dict_callback_config" in dict_parameters.keys()

    dict_pde_config = dict_parameters["dict_pde_config"]
    dict_callback_config = dict_parameters["dict_callback_config"]
    dict_paraview_callback = dict_callback_config["dict_paraview_callback"]
    dict_log_callback = dict_callback_config["dict_log_callback"]

    hydrogym_flow = utils.get_hydrogym_flow(
        name_flow=name_flow)
    path_initial_vectorfield = utils.get_path_initial_vectorfield(
        name_flow=name_flow)
    dt = dict_pde_config["dt"]
    interval_paraview = dict_paraview_callback["interval"]
    interval_log = dict_log_callback["interval"]

    # Create callbacks directory
    path_callbacks = pathlib.Path(f"{path_output_data}/callbacks")
    path_callbacks.mkdir()

    hydrogym_paraview_callback = utils.get_hydrogym_paraview_callback(
        name_flow,
        path_callbacks,
        interval_paraview)
    hydrogym_log_callback = utils.get_hydrogym_log_callback(
        name_flow,
        path_callbacks,
        interval_log)

    env_config = {
        "flow": hydrogym_flow,
        "flow_config": {
            "restart": path_initial_vectorfield,
            "mesh": "coarse",
        },
        "solver": hgym.IPCS,
        "solver_config": {
            "dt": dt,
        },
        "callbacks": [
            hydrogym_paraview_callback,
            hydrogym_log_callback
        ],
    }

    return env_config


def add_attributes(gym_env: hgym.FlowEnv):
    # This is need for now as Firedrake does not propagate dt to the flow
    # object which is passed to the callback
    setattr(gym_env.flow, "dt", gym_env.solver.dt)


# noinspection DuplicatedCode
def integrate_no_control(gym_env: hgym.FlowEnv,
                         n_steps: int | float):
    assert (float(n_steps)).is_integer(), "n_steps must be an integer"
    n_steps = int(n_steps)
    array_action = np.zeros((1, 1))
    for i in range(n_steps):
        gym_env.step(array_action)


def main():
    path_current_script = pathlib.Path(__file__).parent
    path_project_root = path_current_script.parent.parent.parent
    path_mlflow_uri = pathlib.Path(
        path_project_root / "data" / "mlruns").resolve()

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", type=str,
                        help="The config for integration.")
    dict_args = parser.parse_args()

    if dict_args.yaml is None:
        raise ValueError("The config file is required.")

    path_config = dict_args.yaml

    with open(path_config, "r") as file:
        dict_config = yaml.safe_load(file)

    dict_env = dict_config["environment"]

    # Mlflow set-up
    name_xp = dict_config["xp_name"]
    mlflow.set_tracking_uri(f"file:{path_mlflow_uri}")
    mlflow.set_experiment(name_xp)
    experiment = mlflow.get_experiment_by_name(name_xp)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        # MlFlow logging, formatting names of params
        print(f"Starting run {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")

        # Create the path and directory where to store stable_baselines_3 data
        path_output_data = (f"{path_mlflow_uri}/{run.info.experiment_id}/"
                            f"{run.info.run_id}/output_data")
        # noinspection DuplicatedCode
        pathlib.Path(path_output_data).mkdir()

        dict_flattened = {key.split('.')[-1]: value for key, value in
                          pd.json_normalize(dict_config).to_dict(
                              orient='records')[0].items()}
        for key, value in dict_flattened.items():
            mlflow.log_param(key, value)

        # MLFlow logging.
        mlflow.log_param("_id", run.info.run_id)
        mlflow.log_artifact(path_config, artifact_path="config")

        # Get environment name
        env_name = dict_env["name"]

        # Seed setting
        seed = dict_config["seed"]
        random.seed(seed)
        np.random.seed(seed)

        # Create hydrogym config
        dict_hydrogym_config = create_hydrogym_dict_config(
            name_flow=env_name,
            dict_parameters=dict_env["parameters"],
            path_output_data=path_output_data)

        # Create hydrogym environment
        gym_env = hgym.FlowEnv(dict_hydrogym_config)

        # Add attributes to the environment for logging as Firedrake
        # does not propagete enough information to the flow attribute for the
        # logging callback
        add_attributes(gym_env=gym_env)

        # Extract the config for the environment
        dict_pde_config = dict_env["parameters"]["dict_pde_config"]
        n_steps = dict_pde_config["n_steps"]
        # Integrate
        integrate_no_control(gym_env=gym_env,
                             n_steps=n_steps)


if __name__ == "__main__":
    main()
