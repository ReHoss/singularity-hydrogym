from singularity_hydrogym.utils import utils
from singularity_hydrogym.integration import steady_state
import pathlib

LIST_ENVIRONMENTS = ["cylinder", "pinball", "cavity"]
LIST_MESHES = ["medium"]
# Reynolds for pinball include Luc Pastur paper choices

DICT_LIST_REYNOLDS = {
    "cylinder": [10, 30, 50, 75, 90, 105, 120, 130],
    "pinball": [10, 30, 50, 75, 90, 105, 120, 130],
    "cavity": [10, 100, 500, 1000, 2000, 4000, 5000, 7500],
    # "cavity": [5000, 7500],
    # "pinball": [],
    # "cylinder": [],
}

if __name__ == "__main__":
    # Generate the dictionary of parameters
    def main():
        stabilization = "none"
        solver_parameters = {"snes_monitor": None}
        list_dictionary_parameters = [
            {
                "name_flow": name_flow,
                "reynolds": reynolds,
                "name_mesh_resolution": name_mesh_resolution,
                "stabilization": stabilization,
                "solver_parameters": solver_parameters,
            }
            for name_mesh_resolution in LIST_MESHES
            for name_flow in LIST_ENVIRONMENTS
            for reynolds in DICT_LIST_REYNOLDS[name_flow]
        ]

        for dictionary_parameters in list_dictionary_parameters:
            name_flow = dictionary_parameters["name_flow"]
            reynolds_number = dictionary_parameters["reynolds"]
            name_mesh_resolution = dictionary_parameters["name_mesh_resolution"]
            stabilization = dictionary_parameters["stabilization"]
            solver_parameters = dictionary_parameters["solver_parameters"]

            # There is no medium mesh for the pinball, so we skip it
            if name_flow == "pinball" and name_mesh_resolution == "medium":
                continue

            print(
                f"Computing steady state for {name_flow} at Re={reynolds_number} "
                f"with mesh resolution {name_mesh_resolution} and stabilization {stabilization}"
            )

            # Output directory
            name_directory = "_".join(
                [name_flow, str(reynolds_number), name_mesh_resolution, stabilization]
            )
            path_output_data = (
                f"{utils.PATH_PROJECT_ROOT}/data/steady_state"
                f"/{name_flow}/{name_directory}"
            )
            # Create output directory with Pathlib
            pathlib.Path(path_output_data).mkdir(parents=True, exist_ok=True)

            steady_state.compute_steady_state(
                name_flow=name_flow,
                reynolds_number=reynolds_number,
                name_mesh_resolution=name_mesh_resolution,
                stabilization=stabilization,
                solver_parameters=solver_parameters,
                path_output_data=path_output_data,
            )

    main()
