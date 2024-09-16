import tempfile
from typing import Optional

import hydrogym.core
import numpy as np

from singularity_hydrogym.environments import navierstokes2d
from singularity_hydrogym.utils import utils
import firedrake  # pyright: ignore [reportMissingImports]
import pathlib

LIST_ENVIRONMENTS = ["cylinder", "pinball", "cavity"]
LIST_MESHES = ["fine"]
# Reynolds for pinball include Luc Pastur paper choices

DICT_LIST_REYNOLDS = {
    "cylinder": [10, 30, 50, 75, 90, 105, 120, 130],
    "pinball": [10, 30, 50, 75, 90, 105, 120, 130],
    "cavity": [10, 100, 500, 1000, 2000, 4000, 5000, 7500],
}

DICT_DT = {
    "cylinder": 0.01,
    "pinball": 0.01,
    "cavity": 0.001,
}

DICT_DURATION_SECONDS = {
    "cylinder": 0.05,
    "pinball": 0.05,
    "cavity": 0.05,
}


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


if __name__ == "__main__":
    # Generate the dictionary of parameters
    def main():
        stabilization = "none"
        list_dictionary_parameters = [
            {
                "seed": 0,
                "xp_name": "generate_natural_states",
                "environment": {
                    "name": name_flow,
                    "parameters": {
                        "dict_pde_config": {
                            "dt": DICT_DT[name_flow],
                            "reynolds": reynolds,
                            "max_control": 1.0,
                            "control_penalty": 0.0,
                            "interdecision_time_dist": "constant",
                            "mesh": name_mesh_resolution,
                            "actuator_integration": "explicit",
                            "dict_initial_condition": {
                                "type": "equilibrium",
                                "std": 0.1,
                            },
                        },
                        "dict_solver": {
                            "name": "semi_implicit_bdf",
                            "dt": DICT_DT[name_flow],
                            "order": 3,
                            "stabilization": stabilization,
                        },
                        "dict_callback_config": {
                            "dict_paraview_callback": {
                                "interval": 10,
                            },
                            "dict_log_callback": {
                                "interval": 10,
                            },
                        },
                    },
                },
            }
            for name_mesh_resolution in LIST_MESHES
            for name_flow in LIST_ENVIRONMENTS
            for reynolds in DICT_LIST_REYNOLDS[name_flow]
        ]  # TODO: It is also possible to write the config
        #  in a way that pass directly into the constructor without processing

        for dictionary_parameters in list_dictionary_parameters:
            dict_env = dictionary_parameters["environment"]
            dict_pde_config = dict_env["parameters"]["dict_pde_config"]
            name_flow = dict_env["name"]
            reynolds_number = dict_pde_config["reynolds"]
            name_mesh_resolution = dict_pde_config["mesh"]
            stabilization = dict_env["parameters"]["dict_solver"]["stabilization"]

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
                dict_yaml_config=dictionary_parameters,
                path_output_data=path_output_data,
            )

            # Create hydrogym environment
            gym_env = navierstokes2d.NavierStokesFlow2D(**dict_hydrogym_config)

            # Step for the given duration
            duration_seconds = DICT_DURATION_SECONDS[name_flow]
            dt = DICT_DT[name_flow]
            n_steps = compute_steps_from_seconds(duration_seconds, dt)
            step_environment(gym_env, n_steps)

            # Write the flow field to the right directory
            # noinspection PyTypeChecker
            write_flow(
                gym_env.flow,  # pyright: ignore [reportArgumentType]
                reynolds_number,
                path_output_data,
            )

    main()
