import gc
from typing import Optional

import pathlib

import tempfile
import firedrake as fd  # pyright: ignore [reportMissingImports]
import hydrogym.firedrake as hgym

from singularity_hydrogym.utils import utils


def reynolds_curriculum(
    name_flow: str,
    reynolds_number: int,
) -> list[int]:
    assert name_flow in utils.LIST_STR_FLOWS, "Invalid flow name"
    assert reynolds_number > 0, "Reynolds number must be positive"

    list_reynolds_cavity = [500, 1000, 2000, 4000, 7500]
    list_reynolds_pinball = [60, 80, 90, 100, 120, 130]
    list_reynolds_cylinder = [50, 100, 120, 130]

    if name_flow == "cavity" and reynolds_number > min(list_reynolds_cavity):
        # Extract sub-list of Reynolds numbers
        list_curriculum = [
            reynolds for reynolds in list_reynolds_cavity if reynolds < reynolds_number
        ]
        list_curriculum.append(reynolds_number)
    elif name_flow == "pinball" and reynolds_number > min(list_reynolds_pinball):
        # Extract sub-list of Reynolds numbers
        list_curriculum = [
            reynolds for reynolds in list_reynolds_pinball if reynolds < reynolds_number
        ]
        list_curriculum.append(reynolds_number)
    elif name_flow == "cylinder" and reynolds_number > min(list_reynolds_cylinder):
        # Extract sub-list of Reynolds numbers
        list_curriculum = [
            reynolds
            for reynolds in list_reynolds_cylinder
            if reynolds < reynolds_number
        ]
        list_curriculum.append(reynolds_number)
    else:
        list_curriculum = [reynolds_number]

    return list_curriculum


def compute_steady_state(
    name_flow: str,
    reynolds_number: int,
    name_mesh_resolution: str,
    stabilization: str = "none",
    solver_parameters=None,
    path_output_data: Optional[str] = None,
):
    if solver_parameters is None:
        solver_parameters = {}
    assert stabilization in [
        "none",
        "gls",
    ], "Invalid stabilization method"  # TODO: Augment stabilization methods
    assert name_mesh_resolution in utils.LIST_MESHES, "Invalid mesh resolution"
    assert reynolds_number > 0, "Reynolds number must be positive"
    assert name_flow in utils.LIST_STR_FLOWS, "Invalid flow name"

    flow: hgym.FlowConfig
    flow = utils.get_hydrogym_flow(name_flow)(
        Re=reynolds_number, mesh=name_mesh_resolution
    )

    # Generate
    solver = hgym.NewtonSolver(
        flow=flow,
        stabilization=stabilization,  # pyright: ignore [reportCallIssue]
        solver_parameters=solver_parameters,
    )

    # Degree of freedom
    dof: int = flow.mixed_space.dim()
    hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof / fd.COMM_WORLD.size)}")

    # Get the Reynolds curriculum
    list_reynolds_curriculum = reynolds_curriculum(
        name_flow=name_flow, reynolds_number=reynolds_number
    )

    for reynolds_number in list_reynolds_curriculum:
        flow.Re.assign(reynolds_number)
        hgym.print(f"Steady solve at Re={reynolds_number}")
        solver.solve()

    pressure = flow.p
    velocity = flow.u
    vorticity = flow.vorticity()

    # if path_output_data is not None create a temporary directory
    # if path_output_data is None:
    #     path_output_data = tempfile.TemporaryDirectory().name

    if path_output_data is not None:
        # noinspection PyTypeChecker
        # Create output directory with Pathlib
        pathlib.Path(path_output_data).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpfile:
        if path_output_data is None:
            path_output_data = tmpfile
        # Save the solution
        flow.save_checkpoint(f"{path_output_data}/{reynolds_number}_steady.h5")
        # noinspection PyUnresolvedReferences
        pvd = fd.output.VTKFile(f"{path_output_data}/{reynolds_number}_steady.pvd")
        pvd.write(velocity, pressure, vorticity)

    # Garbage collection etc.
    del flow
    del solver
    gc.collect()


if __name__ == "__main__":

    def main():
        name_flow = "cavity"
        reynolds_number = 10
        name_mesh_resolution = "coarse"
        stabilization = "none"
        solver_parameters = {"snes_monitor": None}

        # Output directory
        # name_directory = "_".join(
        #     [name_flow, str(reynolds_number), name_mesh_resolution,
        #      stabilization])
        # path_output_data = (f"{utils.PATH_PROJECT_ROOT}/data/steady_state"
        #                     f"/{name_flow}/{name_directory}")

        path_output_data = None

        # Create output directory with Pathlib
        if path_output_data is not None:
            # noinspection PyTypeChecker
            pathlib.Path(path_output_data).mkdir(parents=True, exist_ok=True)

        compute_steady_state(
            name_flow=name_flow,
            reynolds_number=reynolds_number,
            name_mesh_resolution=name_mesh_resolution,
            stabilization=stabilization,
            solver_parameters=solver_parameters,
            path_output_data=path_output_data,
        )

    main()
