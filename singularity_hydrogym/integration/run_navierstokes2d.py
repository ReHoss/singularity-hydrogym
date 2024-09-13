import argparse
import pathlib
import random
import numpy as np
import mlflow
import pandas as pd
import yaml

import hydrogym.firedrake as hgym
from singularity_hydrogym.environments import navierstokes2d
from singularity_hydrogym.utils import utils


def add_attributes(gym_env: hgym.FlowEnv) -> None:
    # This is need for now as Firedrake does not propagate dt to the flow
    # object which is passed to the callback
    setattr(gym_env.flow, "dt", gym_env.solver.dt)


# noinspection DuplicatedCode
def integrate_no_control(gym_env: hgym.FlowEnv, n_steps: int | float) -> None:
    assert (float(n_steps)).is_integer(), "n_steps must be an integer"
    n_steps = int(n_steps)
    # Derive a null action from the action space
    array_action = gym_env.action_space.sample() * 0.0
    for i in range(n_steps):
        gym_env.step(array_action)


def main() -> None:
    path_current_script = pathlib.Path(__file__)
    path_project_root = path_current_script.parent.parent.parent
    path_mlflow_uri = path_project_root / "data" / "mlruns"

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", type=str, help="The config for integration.")
    dict_args = parser.parse_args()

    if dict_args.yaml is None:
        raise ValueError("The config file is required.")

    path_config = dict_args.yaml

    with open(path_config, "r") as file:
        dict_config = yaml.safe_load(file)

    # Mlflow set-up
    name_xp = dict_config["xp_name"]
    mlflow.set_tracking_uri(f"file:{path_mlflow_uri}")
    mlflow.set_experiment(name_xp)
    experiment = mlflow.get_experiment_by_name(name_xp)
    ml_flow_experiment_id: str = experiment.experiment_id  # pyright: ignore
    with mlflow.start_run(experiment_id=ml_flow_experiment_id) as run:
        # MlFlow logging, formatting names of params
        print(f"Starting run {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")

        # Create the path and directory where to store output data
        path_output_data = (
            f"{path_mlflow_uri}/{run.info.experiment_id}/"
            f"{run.info.run_id}/output_data"
        )
        # noinspection DuplicatedCode
        pathlib.Path(path_output_data).mkdir()

        dict_flattened = {
            key.split(".")[-1]: value
            for key, value in (
                pd.json_normalize(dict_config).to_dict(orient="records")[0].items()
            )
        }
        for key, value in dict_flattened.items():
            mlflow.log_param(key, value)

        # MLFlow logging.
        mlflow.log_param("_id", run.info.run_id)
        mlflow.log_artifact(path_config, artifact_path="config")

        # Seed setting
        seed = dict_config["seed"]
        random.seed(seed)
        np.random.seed(seed)

        # Create hydrogym config
        dict_hydrogym_config = utils.create_navierstokes2d_dict_config(
            dict_yaml_config=dict_config,
            path_output_data=path_output_data,
        )

        # Create hydrogym environment
        gym_env = navierstokes2d.NavierStokesFlow2D(**dict_hydrogym_config)

        # Extract the config for the environment
        n_steps = dict_config["n_steps"]
        # Integrate
        integrate_no_control(gym_env=gym_env, n_steps=n_steps)


if __name__ == "__main__":
    main()
