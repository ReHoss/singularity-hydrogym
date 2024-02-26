import pathlib
import os

import hydrogym

from src.integration import firedrake_evaluate

from typing import Tuple, Callable

PATH_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
LIST_STR_FLOWS = ["cylinder", "pinball", "cavity", "backwardfacingstep"]


def get_hydrogym_flow(name_flow: str):
    assert name_flow in LIST_STR_FLOWS, "Invalid flow name"

    if name_flow == "cylinder":
        return hydrogym.firedrake.Cylinder
    elif name_flow == "pinball":
        return hydrogym.firedrake.Pinball
    elif name_flow == "cavity":
        return hydrogym.firedrake.Cavity
    elif name_flow == "backwardfacingstep":
        return hydrogym.firedrake.Step
    else:
        raise ValueError("Invalid flow name")


def get_path_initial_vectorfield(name_flow: str) -> str:
    assert name_flow in LIST_STR_FLOWS, "Invalid flow name"
    if name_flow == "cavity":
        return (f"{PATH_PROJECT_ROOT}/data/cavity/initial_vector_field/"
                f"reynolds-7500_mesh-coarse_checkpoint.h5")
    else:
        raise NotImplementedError("This flow does not have"
                                  " an initial vectorfield yet")


def get_hydrogym_paraview_callback(name_flow: str,
                                   path_callbacks: str | os.PathLike,
                                   interval: int):
    assert name_flow in LIST_STR_FLOWS, "Invalid flow name"

    if name_flow == "cavity":
        function_firedrake_postprocess = get_firedrake_postprocess(name_flow)[0]
        # noinspection PyUnresolvedReferences
        hydrogym_paraview_callback = hydrogym.firedrake.io.ParaviewCallback(
            interval=interval,
            filename=f"{path_callbacks}/paraview_callback.pvd",
            postprocess=function_firedrake_postprocess
        )
    else:
        raise NotImplementedError("This flow does not have"
                                  " a paraview callback yet")
    return hydrogym_paraview_callback


def get_hydrogym_log_callback(name_flow: str,
                              path_callbacks: str | os.PathLike,
                              interval: int):
    assert name_flow in LIST_STR_FLOWS, "Invalid flow name"

    if name_flow == "cavity":
        function_firedrake_postprocess = get_firedrake_postprocess(name_flow)[1]
        # noinspection PyUnresolvedReferences
        str_log_callback = ("t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t"
                            " KE: {2:0.12e}\t\t TKE: {3:0.12e}")
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
        return (firedrake_evaluate.compute_vorticity,
                firedrake_evaluate.postprocess_cavity)
    else:
        raise NotImplementedError("This flow does not have a postprocess yet")
