import pathlib
import os

import hydrogym
from hydrogym import firedrake as hgym

from src.integration import firedrake_evaluate

from typing import Tuple, Callable, Type

PATH_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
LIST_STR_FLOWS = ["cylinder", "pinball", "cavity", "backward_facing_step"]
LIST_STR_SOLVERS = ["IPCS"]


def get_solver(name_solver: str) -> Type[hgym.solver.TransientSolver]:
    assert name_solver in ["IPCS"], "Invalid solver name"
    if name_solver == "IPCS":
        return hgym.IPCS
    else:
        raise ValueError("Invalid solver name")


def get_hydrogym_flow(name_flow: str) -> Type[hydrogym.firedrake.FlowConfig]:
    assert name_flow in LIST_STR_FLOWS, "Invalid flow name"

    if name_flow == "cylinder":
        return hydrogym.firedrake.Cylinder
    elif name_flow == "pinball":
        return hydrogym.firedrake.Pinball
    elif name_flow == "cavity":
        return hydrogym.firedrake.Cavity
    elif name_flow == "backward_facing_step":
        return hydrogym.firedrake.Step
    else:
        raise ValueError("Invalid flow name")


def get_path_initial_vectorfield(name_flow: str) -> str:
    assert name_flow in LIST_STR_FLOWS, "Invalid flow name"
    if name_flow == "cavity":
        return (
            f"{PATH_PROJECT_ROOT}/data/initial_vector_field/cavity/"
            f"reynolds-7500_mesh-coarse_checkpoint.h5"
        )
    else:
        raise NotImplementedError(
            "This flow does not have" " an initial vectorfield yet"
        )


def get_hydrogym_paraview_callback(
    name_flow: str, path_callbacks: str | os.PathLike, interval: int
) -> hydrogym.firedrake.io.ParaviewCallback:
    assert name_flow in LIST_STR_FLOWS, "Invalid flow name"

    if name_flow == "cavity":
        function_firedrake_postprocess = get_firedrake_postprocess(name_flow)[0]
        # noinspection PyUnresolvedReferences
        hydrogym_paraview_callback = hydrogym.firedrake.io.ParaviewCallback(
            interval=interval,
            filename=f"{path_callbacks}/paraview_callback.pvd",
            postprocess=function_firedrake_postprocess,
        )
    else:
        raise NotImplementedError("This flow does not have" " a paraview callback yet")
    return hydrogym_paraview_callback


def get_hydrogym_log_callback(
    name_flow: str, path_callbacks: str | os.PathLike, interval: int
) -> hydrogym.firedrake.io.LogCallback:
    assert name_flow in LIST_STR_FLOWS, "Invalid flow name"

    if name_flow == "cavity":
        function_firedrake_postprocess = get_firedrake_postprocess(name_flow)[1]
        # noinspection PyUnresolvedReferences
        str_log_callback = (
            "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t" " KE: {2:0.12e}\t\t TKE: {3:0.12e}"
        )
        # noinspection PyUnresolvedReferences
        hydrogym_log_callback = hydrogym.firedrake.io.LogCallback(
            nvals=3,
            interval=interval,
            filename=f"{path_callbacks}/log_callback.dat",  # TODO: .txt?
            print_fmt=str_log_callback,
            postprocess=function_firedrake_postprocess,
        )
    else:
        raise NotImplementedError("This flow does not have a log callback yet")
    return hydrogym_log_callback


def get_firedrake_postprocess(name_flow: str) -> Tuple[Callable, Callable]:
    assert name_flow in LIST_STR_FLOWS, "Invalid flow name"
    if name_flow == "cavity":
        return (
            firedrake_evaluate.compute_vorticity,
            firedrake_evaluate.postprocess_cavity,
        )
    else:
        raise NotImplementedError("This flow does not have a postprocess yet")


def create_hydrogym_dict_config(
    name_flow: str, dict_parameters: dict, path_output_data: str
) -> dict:
    assert name_flow in LIST_STR_FLOWS, "Invalid flow name"
    assert "dict_pde_config" in dict_parameters.keys()
    assert "dict_callback_config" in dict_parameters.keys()

    dict_pde_config = dict_parameters["dict_pde_config"]
    dict_callback_config = dict_parameters["dict_callback_config"]
    dict_paraview_callback = dict_callback_config["dict_paraview_callback"]
    dict_log_callback = dict_callback_config["dict_log_callback"]

    hydrogym_flow = get_hydrogym_flow(name_flow=name_flow)
    path_initial_vectorfield = get_path_initial_vectorfield(name_flow=name_flow)
    dt = dict_pde_config["dt"]

    interval_paraview = dict_paraview_callback["interval"]
    interval_log = dict_log_callback["interval"]

    # Create callbacks directory
    path_callbacks = pathlib.Path(f"{path_output_data}/callbacks")
    path_callbacks.mkdir()

    hydrogym_paraview_callback = get_hydrogym_paraview_callback(
        name_flow, path_callbacks, interval_paraview
    )
    hydrogym_log_callback = get_hydrogym_log_callback(
        name_flow, path_callbacks, interval_log
    )

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
        "callbacks": [hydrogym_paraview_callback, hydrogym_log_callback],
    }

    return env_config
