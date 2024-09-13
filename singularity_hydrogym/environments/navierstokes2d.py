import firedrake  # pyright: ignore [reportMissingImports]
import pathlib

import logging
import tempfile

import hydrogym
from matplotlib import pyplot as plt
from typing import Optional, Type, Any
import gymnasium
from gymnasium import spaces
from singularity_hydrogym.environments import abstract_interface
from singularity_hydrogym.utils import utils
import numpy as np
import numpy.typing as npt

# TODO: tests


DICT_DEFAULT_INITIAL_CONDITION = {
    "type": "equilibrium",
    "std": 0.1,
}


DICT_DEFAULT_SOLVER = {
    "name": "semi_implicit_bdf",
    "dt": 0.01,
    "order": 2,
    "stabilization": "none",
}


class NavierStokesFlow2D(  # pyright: ignore [reportIncompatibleMethodOverride, reportIncompatibleVariableOverride]
    hydrogym.FlowEnv,
    abstract_interface.EnvInterfaceControlDDE,
):
    name_env = "navier_stokes"

    def __init__(
        self,
        seed: int = 0,
        name_flow: str = "pinball",
        max_control: float = 1.0,
        reynolds: int | float = 30,
        dt: float = 0.001,
        dtype: str = "float32",
        control_penalty: float = 0.0,
        interdecision_time_dist: str = "constant",
        dict_initial_condition: Optional[dict] = None,
        path_hydrogym_checkpoint: Optional[str] = None,
        mesh: str = "coarse",
        actuator_integration: str = "explicit",
        dict_solver: Optional[dict] = None,
        paraview_callback_interval: int = 10,
        log_callback_interval: int = 10,
        path_output_data: Optional[str] = None,
    ) -> None:
        """
        Constructor for the Cavity environment.
        """

        self._check_arguments(
            name_flow=name_flow,
            max_control=max_control,
            reynolds=reynolds,
            dt=dt,
            dtype=dtype,
            control_penalty=control_penalty,
            interdecision_time_dist=interdecision_time_dist,
            dict_initial_condition=dict_initial_condition,
            path_hydrogym_checkpoint=path_hydrogym_checkpoint,
            mesh=mesh,
            actuator_integration=actuator_integration,
            dict_solver=dict_solver,
            paraview_callback_interval=paraview_callback_interval,
            log_callback_interval=log_callback_interval,
            path_output_data=path_output_data,
        )

        # Set solver dictionary
        if dict_solver is None:
            dict_solver = DICT_DEFAULT_SOLVER

        # Get the path of this file to use as the root directory
        self.path_current_file = pathlib.Path(__file__)
        self.name_flow = name_flow

        np_dtype = np.dtype(dtype)
        assert np_dtype.type in [np.float32, np.float64], "Only float32/64 supported"
        self.np_dtype = np_dtype

        self.hydrogym_flow = utils.get_hydrogym_flow(name_flow)
        self.path_initial_vectorfield = utils.get_path_initial_vectorfield(
            name_flow=name_flow
        )
        self.dt = dt

        self.max_control = max_control
        self.control_penalty = control_penalty

        self.solver_dt: float = dict_solver["dt"]
        self.name_solver: str = dict_solver["name"]
        self.solver_order: int = dict_solver["order"]
        self.solver_stabilization: str = dict_solver["stabilization"]
        self.solver_class: Type[hydrogym.core.TransientSolver] = utils.get_solver(
            name_solver=self.name_solver
        )

        self.reynolds = float(reynolds)
        self.mesh = mesh

        self.path_hydrogym_checkpoint = path_hydrogym_checkpoint
        self.actuator_integration = actuator_integration

        self.tempfile_tempdir: Optional[tempfile.TemporaryDirectory] = None
        self.path_output_data = path_output_data
        self.check_or_create_path_output_data()
        # Set up the paraview callback
        self.paraview_callback_interval = paraview_callback_interval
        self.paraview_callback: Optional[hydrogym.firedrake.io.ParaviewCallback]
        self.paraview_callback = self.get_paraview_callback()

        # Set up the log callback
        self.log_callback_interval = log_callback_interval
        self.log_callback: Optional[hydrogym.firedrake.io.LogCallback]
        self.log_callback = self.get_log_callback()

        self.dict_env_config = {
            "flow": self.hydrogym_flow,
            "flow_config": {
                "restart": self.path_initial_vectorfield,
                "mesh": self.mesh,
                # "Re": self.reynolds,
                # "actuator_integration": self.actuator_integration,
            },
            "solver": self.solver_class,
            "solver_config": {
                "dt": self.solver_dt,
                "order": self.solver_order,
                "stabilization": self.solver_stabilization,
            },
            "callbacks": [self.paraview_callback, self.log_callback],
        }

        # Initialise the hydrogym environment
        hydrogym.FlowEnv.__init__(self, env_config=self.dict_env_config)

        # The state space is a 2d continuous domain ...
        # ... with the velocity field and the pressure field
        dim_x = int(self.flow.mesh.cell_set.size)  # TODO: flow.mixed_space.dim() check
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(dim_x, 3),
            dtype=self.np_dtype.type,
        )

        # Patch the spaces to Gymnasium
        self._patch_box_space()
        self._patch_flow_attributes()

        assert isinstance(self.action_space, gymnasium.spaces.Box)
        assert isinstance(self.observation_space, gymnasium.spaces.Box)

        abstract_interface.EnvInterfaceControlDDE.__init__(
            self,
            action_space=self.action_space,
            dt=self.dt,
            dtype=dtype,
            observation_space=self.observation_space,
            state_space=self.state_space,
            seed=seed,
            interdecision_time_dist=interdecision_time_dist,
        )

        # No putting any adaptative stepsize controller even for Dopri5
        # This is a delibarate choice to keep the system simple

        if dict_initial_condition is None:
            self.dict_initial_condition = DICT_DEFAULT_INITIAL_CONDITION
        else:
            self.dict_initial_condition = dict_initial_condition
        assert self.dict_initial_condition["type"] in [
            "equilibrium",
        ], "Initial condition type must be zero or nonzero"

        # The below interface creates the mandatory attributes
        # This forces all environments to have the same interface
        abstract_interface.EnvInterfaceControlDDE.__init__(
            self,
            action_space=self.action_space,
            dt=self.dt,
            dtype=dtype,
            observation_space=self.observation_space,
            state_space=self.state_space,
            seed=seed,
            interdecision_time_dist=interdecision_time_dist,
        )
        # Abstract interface creates attributes such as self.np_dtype

        # Initialize the arrays containing the state, action, and observation
        self._array_state: npt.NDArray | None = None
        self._array_control: npt.NDArray | None = None
        self._array_observation: npt.NDArray
        # Include None in the possible types for the list
        # Pyright complains since the type is redefined
        self._list_array_state_history: list[npt.NDArray | None]  # pyright: ignore [reportIncompatibleVariableOverride]

        self._array_observation, _ = self.reset(seed=seed, options=None)
        # self.state = self.state  # Otherwise, pyright will complain about setting
        # the attribute before defining it in __init__; this flaw is due to the
        # fact PendulumEnv does not have a getter for state

    def __del__(self):
        """
        Destructor takes care of cleaning up the temporary directory.
        """
        if self.tempfile_tempdir is not None:
            self.tempfile_tempdir.cleanup()

    # noinspection PyMethodOverriding
    def reset(  # pyright: ignore [reportIncompatibleMethodOverride]
        self, seed: int, options: Optional[dict] = None
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        # Pyright: Warning ignored because the parent method is not editable
        # since it is an external dep
        """
        Reset the environment to a new initial state.

        Args:
            seed: The seed for the random number generator.
            options: A dictionary of options for the reset.

        Returns:
            The initial observation.
        """  # TODO: Check if the seed propagates well
        abstract_interface.EnvInterfaceControlDDE.reset(
            self, seed=seed, options=options
        )
        t0 = 0.0
        # if options is None:
        #     raise NotImplementedError
        #     if self.dict_initial_condition["type"] == "equilibrium":
        #         self.q0 = self.hydrogym_flow.get_attractor()
        #         self.q0 = self.q0 + np.random.normal(
        #             scale=self.dict_initial_condition["std"],
        #             size=self.q0.shape,
        #         )
        #
        #         array_noise = self.np_random.normal(
        #             loc=0.0, scale=self.dict_initial_condition["std"], size=(1,)
        #         )
        #         self._array_state = array_equilibrium + array_noise
        #         self.construct_constant_history_from_state(
        #             t0=t0, array_state=self._array_state
        #         )
        #
        #
        #
        #     else:
        #         raise ValueError("Invalid initial condition type")
        # elif "state" in options:
        #     raise NotImplementedError
        # else:
        #     raise ValueError("Invalid options")

        # TODO: Implement the noise on the initial condition self.q0 for a proper reset
        # TODO: Check the dtypes
        # Reset the environment

        # --- Hack based exactly on the internal FlowEnv reset method
        # --- only one line is added to customise the flow reset

        # tuple_obs = hydrogym.FlowEnv.reset(self)
        self.iter = 0
        # That updates q0
        self.set_initial_condition()

        # I suppose reset is not needed since the flow is reset
        # in the set_initial_condition through the load_checkpoint
        # self.flow.reset(q0=self.q0)  # pyright: ignore
        self.solver.reset()
        tuple_obs = self.flow.get_observations()

        # --- End of the hack

        # Cast the observation to the right dtype
        # From tuple to array
        self._array_observation = np.array(tuple_obs, dtype=self.np_dtype)
        self._list_time_points = [t0]
        self._list_array_obs_history = [self.array_observation]
        # Pyright type ignore since state is not implemented
        self._list_array_state_history.append(None)
        dict_info = {}
        return self._array_observation, dict_info

    # noinspection PyMethodOverriding
    # Ignored because the parent method is not editable since it is an external dep
    def step(  # pyright: ignore [reportIncompatibleMethodOverride]
        self, action: npt.NDArray[np.floating]
    ) -> tuple[
        npt.NDArray[np.floating], float, bool, bool, dict
    ]:  # TODO: Implement semi-markovian
        """
        Step the environment forward in time.

        Args:
            action: The action to take.

        Returns:
            The new observation, the reward, whether the episode is done, and any additional information.
        """
        # Check if the action is valid
        assert self.action_space.contains(action), "Action not valide (dtype?)."

        # Get the common divisor of the dt and the solver_dt
        # This is a hack to avoid changing the hydrogym interface
        number_of_internal_steps = int(self.dt / self.solver_dt)
        # assert number_of_internal_steps >= 1, "The number of internal steps must be +"
        if number_of_internal_steps < 1:
            raise ValueError("The number of internal steps must be positive")

        tuple_obs: tuple[np.floating, np.floating, np.floating, np.floating]
        reward: float
        done: bool
        dict_info: dict

        for _ in range(number_of_internal_steps - 1):
            hydrogym.FlowEnv.step(self, action)
        tuple_obs, reward, done, dict_info = hydrogym.FlowEnv.step(self, action)  # pyright: ignore [reportAssignmentType]
        # Suppressed above warning due to typing error

        # The obs vector returned has a tuple format
        # Cast the observation to the right dtype
        # From tuple to array
        self._array_observation = np.array(tuple_obs, dtype=self.np_dtype)

        # Check if the observation is valid
        assert self.observation_space.contains(
            self._array_observation
        ), "Observation not valid (dtype?)."

        # Set the dt of the environment only for the step method call
        # Start: That's a hack to avoid changing the Gymnasium interface
        t0: float = self.current_t
        old_dt = self.dt
        dt = self.sample_interdecision_time()

        # Update the time and history
        self._list_time_points.append(t0 + dt)
        self.list_array_obs_history.append(self.array_observation)
        # Pyright type ignore since state is not implemented
        self.list_array_state_history.append(None)  # pyright: ignore [reportArgumentType]

        self.dt = old_dt
        # End: Set the dt back to the original value in order to avoid side effects

        truncated = False
        return self.array_observation, reward, done, truncated, dict_info

    def get_paraview_callback(self) -> hydrogym.firedrake.io.ParaviewCallback | None:
        if self.path_output_data is None:
            raise ValueError("The path_output_data is not provided")
        else:
            assert (
                self.paraview_callback_interval is not None
            ), "Interval must be provided"

        path_callbacks = f"{self.path_output_data}/paraview_callbacks"
        pathlib.Path(path_callbacks).mkdir(parents=True)
        hydrogym_paraview_callback = utils.get_hydrogym_paraview_callback(
            name_flow=self.name_flow,
            path_callbacks=path_callbacks,
            interval=self.paraview_callback_interval,
        )
        return hydrogym_paraview_callback

    def get_log_callback(
        self,
    ) -> hydrogym.firedrake.io.LogCallback | None:
        if self.path_output_data is None:
            raise ValueError("The path_output_data is not provided")

        path_callbacks = f"{self.path_output_data}/log_callbacks"
        pathlib.Path(path_callbacks).mkdir(parents=True)

        hydrogym_log_callback = utils.get_hydrogym_log_callback(
            name_flow=self.name_flow,
            path_callbacks=path_callbacks,
            interval=self.log_callback_interval,
        )
        return hydrogym_log_callback

    def check_or_create_path_output_data(self) -> None:
        if self.path_output_data is None:
            logging.warning(
                "The path_output_data is not provided."
                "Creating a temporary directory."
            )
            # noinspection PyAttributeOutsideInit
            self.tempfile_tempdir = tempfile.TemporaryDirectory()
            self.path_output_data = self.tempfile_tempdir.name
        else:
            # Check if the directory exists with pathlib
            path_output_data = pathlib.Path(self.path_output_data)
            if not path_output_data.exists():
                raise FileNotFoundError(
                    f"The directory {self.path_output_data} does not exist."
                )

    @property
    def array_control(self) -> npt.NDArray[np.floating] | None:
        return self._array_control

    @property
    def array_observation(self) -> npt.NDArray[np.floating]:
        return self._array_observation

    @property
    def array_state(self) -> npt.NDArray[np.floating]:
        # Trigger warning since the state is to complex to be represented
        logging.warning("The array_state property is accessed but not implemented")
        raise NotImplementedError("The state is to complex to be represented")

    @array_state.setter
    def array_state(self, array_x: npt.NDArray[np.floating]) -> None:
        raise NotImplementedError("The state is to complex to be set")

    def cost_function(
        self,
        time: float,
        array_control: npt.NDArray,
        array_observation: npt.NDArray,
        array_observation_next: npt.NDArray,
    ) -> float:
        raise NotImplementedError("The cost function is not implemented yet here")

    def _convert_dtype_gym_spaces_box(
        self, gym_space: gymnasium.spaces.Box
    ) -> gymnasium.spaces.Box:
        """
        Patch the gym.spaces.Box to allow for the dtype attribute.
        """
        gymnasium_space = gymnasium.spaces.Box(
            low=gym_space.low,
            high=gym_space.high,
            shape=gym_space.shape,
            # Notably, the dtype attribute is modified voluntarily here
            dtype=self.np_dtype.type,
        )
        return gymnasium_space

    # noinspection PyTypeChecker
    def _patch_box_space(self) -> None:
        """
        Patch the gym.spaces.Box to allow for the dtype attribute.
        """

        self.state_space: gymnasium.spaces.Box = self._convert_dtype_gym_spaces_box(
            self.state_space
        )
        self.action_space: gymnasium.spaces.Box = self._convert_dtype_gym_spaces_box(
            self.action_space
        )
        self.observation_space: gymnasium.spaces.Box = (
            self._convert_dtype_gym_spaces_box(self.observation_space)
        )

    def _patch_flow_attributes(self) -> None:
        """
        Patch the Flow to add the attributes for logging.
        """
        # Add the attributes to the environment for logging (callback)
        setattr(self.flow, "dt", self.dt)

    def set_initial_condition(self) -> None:
        """
        Set the initial condition of the environment.
        """
        # TODO: self.hydrogym_checkpoint should be an initial condition mode!
        if self.dict_initial_condition["type"] == "equilibrium":
            fd_rng = firedrake.randomfunctiongen.Generator(
                firedrake.randomfunctiongen.PCG64(seed=1234)
            )

            path_file_h5: str = self.get_initial_condition_file()
            noise_std = self.dict_initial_condition["std"]
            self.flow.load_checkpoint(filename=path_file_h5)
            # noinspection PyUnresolvedReferences
            self.flow.q += fd_rng.normal(self.flow.mixed_space, 0.0, noise_std)  # pyright: ignore [reportAttributeAccessIssue]
            self.flow.q0 = self.flow.q  # pyright: ignore [reportAttributeAccessIssue]
            # Warning suppressed since it is wrongly initialized
        else:
            raise ValueError("Invalid initial condition type")

    def get_initial_condition_file(self) -> str:
        """
        Get the initial condition file.
        """
        if self.dict_initial_condition["type"] == "equilibrium":
            path_file_h5 = (
                f"{self.path_current_file.parent}/data/steady_state/{self.name_flow}/"
                f"{self.name_flow}_{int(self.reynolds)}_{self.mesh}_none/"
                f"{int(self.reynolds)}_steady.h5"
            )

            # Check if the file exists
            if not pathlib.Path(path_file_h5).exists():
                raise FileNotFoundError(
                    f"The file {path_file_h5} which contains"
                    f" the initial condition does not exist."
                )
            return path_file_h5

        else:
            raise ValueError("Invalid initial condition type")

    @staticmethod
    def _check_arguments(
        name_flow: str,
        max_control: float,
        reynolds: float,
        dt: float,
        dtype: str,
        control_penalty: float,
        interdecision_time_dist: str,
        dict_initial_condition: Optional[dict],
        path_hydrogym_checkpoint: Optional[str],
        mesh: str,
        actuator_integration: str,
        dict_solver: Optional[dict],
        paraview_callback_interval: int,
        log_callback_interval: int,
        path_output_data: Optional[str],
    ) -> None:
        """
        Check the arguments of the constructor.
        """
        assert isinstance(name_flow, str), "name_flow must be a string"
        assert isinstance(max_control, float), "max_control must be a float"
        assert isinstance(reynolds, (int, float)), "reynolds must be a number"
        assert isinstance(dt, float), "dt must be a float"
        assert dtype in ["float32", "float64"], "dtype must be float32 or float64"
        assert isinstance(control_penalty, float), "control_penalty must be a float"

        assert interdecision_time_dist in [
            "constant",
            "exponential",
        ], "interdecision_time_dist must be constant or exponential"

        if dict_initial_condition is not None:
            assert isinstance(
                dict_initial_condition, dict
            ), "dict_initial_condition must be a dict."

        if path_hydrogym_checkpoint is not None:
            assert isinstance(
                path_hydrogym_checkpoint, str
            ), "path_hydrogym_checkpoint must be a string."
            raise NotImplementedError(
                "The path_hydrogym_checkpoint is not implemented."
            )

        assert mesh in ["coarse", "medium", "fine"], "mesh must be coarse or fine."
        assert actuator_integration in [
            "explicit",
            "implicit",
        ], "actuator_integration must be explicit or implicit."
        assert isinstance(
            paraview_callback_interval, int
        ), "paraview_callback_interval must be an int."
        assert isinstance(
            log_callback_interval, int
        ), "log_callback_interval must be an int."
        if path_output_data is not None:
            assert isinstance(
                path_output_data, str
            ), "path_output_data must be a string."

        if dict_solver is None:
            dict_solver = DICT_DEFAULT_SOLVER
        name_solver = dict_solver["name"]
        solver_dt = dict_solver["dt"]

        assert name_solver in [
            "semi_implicit_bdf",
            "explicit_euler",
        ], "name_solver must be semi_implicit_bdf or explicit_euler."
        assert isinstance(solver_dt, float), "solver_dt must be a float."
        # Both dt should have a common divisor
        assert dt > 0, "dt must be positive."
        assert solver_dt > 0, "solver_dt must be positive."
        assert dt >= solver_dt, "dt must be greater than or equal to solver_dt."
        assert dt % solver_dt == 0, "dt and solver_dt must have a common divisor."


if __name__ == "__main__":

    def main():
        # Test the Cavity environment
        seed = 0
        name_flow = "cavity"
        max_control = 1.0
        reynolds = 7500.0
        dt = 0.001
        dtype = "float32"
        control_penalty = 0.0
        interdecision_time_dist = "constant"
        dict_initial_condition = None
        path_hydrogym_checkpoint = None
        mesh = "coarse"
        actuator_integration = "explicit"
        dict_solver = {
            "name": "semi_implicit_bdf",
            "dt": 0.001,
            "order": 2,
            "stabilization": "none",
        }
        paraview_callback_interval = 10
        log_callback_interval = 10
        path_output_data = None

        env = NavierStokesFlow2D(
            seed=seed,
            name_flow=name_flow,
            max_control=max_control,
            reynolds=reynolds,
            dt=dt,
            dtype=dtype,
            control_penalty=control_penalty,
            interdecision_time_dist=interdecision_time_dist,
            dict_initial_condition=dict_initial_condition,
            path_hydrogym_checkpoint=path_hydrogym_checkpoint,
            mesh=mesh,
            actuator_integration=actuator_integration,
            dict_solver=dict_solver,
            paraview_callback_interval=paraview_callback_interval,
            log_callback_interval=log_callback_interval,
            path_output_data=path_output_data,
        )

        n_steps = 50
        # Generate a trajectory and plot it
        dim_obs = env.observation_space.shape[0]
        trajectory = np.zeros((n_steps, dim_obs))
        for i in range(n_steps):
            assert isinstance(env.action_space, gymnasium.spaces.Box)
            action_ = env.action_space.sample() * 0.0
            # action_ = env.action_space.low * 0.5
            observation, reward_, done_, info_, truncated_ = env.step(action_)
            del reward_, done_, info_, truncated_
            trajectory[i] = observation

        plt.plot(env.array_time_points, env.list_array_obs_history)
        # Write the trajectory to a file
        plt.savefig("/tmp/.trash/trajectory.png")
        plt.show()

    main()
