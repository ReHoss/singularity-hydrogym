# Create a generator of config files for python
import argparse
import datetime
import itertools
import pathlib

import yaml


"""
1 - Do not forget to change the mlfow_experiment_name value
2 - The file naming may impact file generation when overwriting is permitted, an
    exception has been included to prevent this
"""

text_yaml_config_file = """
mlflow_experiment_name: "hydrogym_simulation_cavity_30s_natural_flow"
seed: 0
env:
  name: "cavity"
  parameters:
    dict_pde_config:
      dt: 0.01
      reynolds: 5000
      max_control: 0.0
      control_penalty: 0.0
      interdecision_time_dist: "constant"
      mesh: "coarse"
      actuator_integration: "explicit"
      dict_initial_condition:
        type: "equilibrium"
        std: 0.1
    dict_solver:
      name: "semi_implicit_bdf"
      dt: 0.001
      order: 3
      stabilization: "none"
    dict_callback_config:
      dict_paraview_callback:
        interval: 10
      dict_log_callback:
        interval: 10

duration_seconds: 30.0
"""

# Default target directory with the date and time
# noinspection DuplicatedCode
name_target_directory = (
    f"{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_generated_configs"
)

# Argparse the name of the target directory with flag -d or --directory (optional)
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help="Name of the target directory")
args = parser.parse_args()
name_target_directory = args.directory if args.directory else name_target_directory

# Get current file directory with Pathlib
path_parent_directory = pathlib.Path(__file__).parent

path_target_directory = pathlib.Path(
    f"{path_parent_directory}/../../../configs/batch/{name_target_directory}"
).resolve()
# Check if the target directory exists
if not pathlib.Path(pathlib.Path(path_target_directory).resolve()).exists():
    # Create the target directory
    pathlib.Path(path_target_directory).mkdir(parents=False, exist_ok=False)

# Load the yaml text as a dictionary
dict_config = yaml.load(text_yaml_config_file, Loader=yaml.FullLoader)

# Add a suffix to the mlflow_experiment_name to differentiate the runs
current_time = datetime.datetime.now().strftime("%Hh_%Mm")
dict_config["mlflow_experiment_name"] += f"_{current_time}"

# --- Start of the list of parameters to change
# Define the list of parameters to change,
# by giving in the tuple the nested keys of the dictionary

N_SEEDS = 1
LIST_REYNOLDS = [10, 100, 500, 1000, 2000, 4000, 5000, 7500]

list_reynolds = [
    ("env", "parameters", "dict_pde_config", "reynolds", reynolds)
    for reynolds in LIST_REYNOLDS
]

list_seed = [("seed", seed) for seed in range(N_SEEDS)]

# !!!! UPDATE THE NESTED LIST OF PARAMETERS TO CHANGE HERE !!!!
nested_list_parameters = [list_seed, list_reynolds]

# --- End of the list of parameters to change


for tuple_parameters in itertools.product(*nested_list_parameters):
    # Create a new dictionary with the new parameters
    dict_config_new = dict_config.copy()
    for tuple_parameter in tuple_parameters:
        nested_keys = tuple_parameter[:-1]
        value = tuple_parameter[-1]
        # Modify the nested dictionary at the given keys,
        # access the nested dictionary directly
        nested_dict = dict_config_new
        for key in nested_keys[:-1]:
            nested_dict = nested_dict[key]
        nested_dict[nested_keys[-1]] = value

    # Create the name of the config file
    env_name = dict_config_new["env"]["name"]

    seed = dict_config_new["seed"]
    name_env = dict_config_new["env"]["name"]
    duration_seconds = dict_config_new["duration_seconds"]
    reynolds = dict_config_new["env"]["parameters"]["dict_pde_config"]["reynolds"]

    name_config_file = "".join(
        f"env_name_{name_env}"
        f"_seed_{seed}"
        f"_duration_{duration_seconds}"
        f"_reynolds_{reynolds}"
        ".yaml"
    )

    # Create the path of the config file
    path_config_file = f"{path_target_directory}/{name_config_file}"
    # Check if the config file exists
    if pathlib.Path(pathlib.Path(path_config_file).resolve()).exists():
        raise FileExistsError(f"The file {path_config_file} already exists")
    # Write the config file
    with open(path_config_file, "w") as file:
        yaml.dump(dict_config_new, file, default_flow_style=False, sort_keys=False)
