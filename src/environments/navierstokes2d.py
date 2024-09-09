import gym.spaces
import pathlib

import logging
import tempfile

import hydrogym
from matplotlib import pyplot as plt
from typing import Optional, Type, Any
import gymnasium
from gymnasium import spaces
from src.environments import abstract_interface
from src.utils import utils
import numpy as np
import numpy.typing as npt

# TODO: temporary path
# TODO: tests


DICT_DEFAULT_INITIAL_CONDITION = {
    "type": "fixed_on_attractor",
    "std": 0.1,
}


class NavierStokesFlow2D(  # pyright: ignore [reportIncompatibleMethodOverride, reportIncompatibleVariableOverride]
    hydrogym.FlowEnv,
    abstract_interface.EnvInterfaceControlDDE,
):
    name_env = "navier_stokes"

    def __init__(
        self,
        seed: int,
        name_flow: str = "cylinder",
        max_control: float = 1.0,
        reynolds: float = 7500.0,
        dt: float = 0.0001,
        dtype: str = "float32",
        control_penalty: float = 0.0,
        interdecision_time_dist: str = "constant",
        dict_initial_condition: Optional[dict] = None,
        path_hydrogym_checkpoint: Optional[str] = None,
        mesh: str = "coarse",
        actuator_integration: str = "explicit",
        name_solver: str = "IPCS",
        paraview_callback_interval: int = 10,
        log_callback_interval: int = 10,
        path_output_data: Optional[str] = None,
    ) -> None:
        """
        Constructor for the Cavity environment.
        """

        # self._check_arguments(  # TODO: Check that
        #     max_control=max_control,
        #     reynolds=reynolds,
        #     dt=dt,
        #     dtype=dtype,
        #     control_penalty=control_penalty,
        #     seed=seed,
        #     interdecision_time_dist=interdecision_time_dist,
        #     dict_initial_condition=dict_initial_condition,
        # )

        self.name_flow = name_flow

        np_dtype = np.dtype(dtype)
        assert np_dtype.type in [np.float32, np.float64], "Only float32/64 supported"
        assert isinstance(np_dtype, np.floating)
        self.np_dtype = np_dtype

        self.hydrogym_flow = utils.get_hydrogym_flow(name_flow)
        self.path_initial_vectorfield = utils.get_path_initial_vectorfield(
            name_flow=name_flow
        )
        self.dt = dt
        self.max_control = max_control

        self.control_penalty = control_penalty
        self.name_solver = name_solver
        self.solver_class: Type[
            hydrogym.firedrake.solver.TransientSolver
        ] = utils.get_solver(name_solver=self.name_solver)  # Hardcoded

        self.reynolds = reynolds
        self.mesh = mesh

        self.path_hydrogym_checkpoint = path_hydrogym_checkpoint
        self.actuator_integration = actuator_integration

        self.path_output_data = (
            path_output_data  # TODO: create temp directory for output data
        )
        self.check_or_create_path_output_data()
        # Set up the paraview callback
        self.paraview_callback_interval = paraview_callback_interval
        self.paraview_callback: Optional[hydrogym.firedrake.io.ParaviewCallback]
        self.paraview_callback = self.get_paraview_callback()

        # Set up the log callback
        self.log_callback_interval = log_callback_interval
        self.log_callback: Optional[hydrogym.firedrake.io.LogCallback]
        self.log_callback = self.get_log_callback()

        # TODO: Clarify thing with the add_attributes
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
                "dt": self.dt,
            },
            "callbacks": [self.paraview_callback, self.log_callback],
        }

        # Initialise the hydrogym environment
        hydrogym.FlowEnv.__init__(self, env_config=self.dict_env_config)

        # The state space is a 2d continuous domain ...
        # ... with the velocity field and the pressure field
        dim_x = int(self.flow.mesh.cell_set.size)
        self.state_space = spaces.Box(  # TODO: Get the mesh size
            low=-np.inf,
            high=np.inf,
            shape=(dim_x, 3),
            dtype=self.np_dtype,
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
            "fixed_on_attractor",
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
        #     if self.dict_initial_condition["type"] == "fixed_on_attractor":
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
        tuple_obs = hydrogym.FlowEnv.reset(self)
        # Cast the observation to the right dtype
        # From tuple to array
        self._array_observation = np.array(tuple_obs, dtype=self.np_dtype)
        self._list_time_points.append(t0)
        self._list_array_obs_history.append(self.array_observation)
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
    ]:  # TODO: Check the dt behavior
        """
        Step the environment forward in time.

        Args:
            action: The action to take.

        Returns:
            The new observation, the reward, whether the episode is done, and any additional information.
        """
        # Check if the action is valid
        assert self.action_space.contains(action), "Action not valide (dtype?)."
        # The obs vector returned has a tuple format
        tuple_obs, reward, done, dict_info = hydrogym.FlowEnv.step(self, action)
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
        )  # TODO: Fix path
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
            self.path_output_data = tempfile.TemporaryDirectory().name
            # TODO: Check the path_output_data
        else:
            # Check if the directory exists with pathlib
            path_output_data = pathlib.Path(self.path_output_data)
            path_output_data.mkdir(exist_ok=False)

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

    def _convert_gym_spaces_box_gymnasium(
        self, gym_space: gym.spaces.Box
    ) -> gymnasium.spaces.Box:
        """
        Patch the gym.spaces.Box to allow for the dtype attribute.
        """
        gymnasium_space = gymnasium.spaces.Box(
            low=gym_space.low,
            high=gym_space.high,
            shape=gym_space.shape,
            # Notably, the dtype attribute is modified voluntarily here
            dtype=self.np_dtype,
        )
        return gymnasium_space

    # noinspection PyTypeChecker
    def _patch_box_space(self) -> None:
        """
        Patch the gym.spaces.Box to allow for the dtype attribute.
        """
        assert isinstance(self.state_space, gym.spaces.Box)
        assert isinstance(self.action_space, gym.spaces.Box)
        assert isinstance(self.observation_space, gym.spaces.Box)

        self.state_space: gymnasium.spaces.Box = self._convert_gym_spaces_box_gymnasium(
            self.state_space
        )
        self.action_space: gymnasium.spaces.Box = (
            self._convert_gym_spaces_box_gymnasium(self.action_space)
        )
        self.observation_space: gymnasium.spaces.Box = (
            self._convert_gym_spaces_box_gymnasium(self.observation_space)
        )

    def _patch_flow_attributes(self) -> None:
        """
        Patch the Flow to add the attributes for logging.
        """
        # Add the attributes to the environment for logging (callback)
        setattr(self.flow, "dt", self.dt)


if __name__ == "__main__":

    def main():
        # Test the Cavity environment
        seed = 0
        name_flow = "cavity"
        max_control = 1.0
        reynolds = 7500.0
        dt = 0.0001
        dtype = "float32"
        control_penalty = 0.0
        interdecision_time_dist = "constant"
        dict_initial_condition = None
        path_hydrogym_checkpoint = None
        mesh = "coarse"
        actuator_integration = "explicit"
        name_solver = "IPCS"
        paraview_callback_interval = 10000
        log_callback_interval = 100
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
            name_solver=name_solver,
            paraview_callback_interval=paraview_callback_interval,
            log_callback_interval=log_callback_interval,
            path_output_data=path_output_data,
        )

        n_steps = 10000
        # Generate a trajectory and plot it
        trajectory = np.zeros((n_steps, 1))
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
