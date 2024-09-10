from abc import abstractmethod
from typing import ClassVar

import gymnasium
import numpy as np
from gymnasium import Env
from numpy import typing as npt


class EnvInterfaceControlDDE(gymnasium.Env):
    """
    This class defines the common interface for all environments
     in the control_dde library.
    """

    # state_space: gymnasium.spaces.Box
    # action_space: gymnasium.spaces.Box
    # observation_space: gymnasium.spaces.Box
    name_env: ClassVar[str]

    def __new__(cls, *args, **kwargs):
        """Specially designed to check existence of class attributes."""
        del args, kwargs
        # Check emptiness of name_env !
        if not cls.name_env:
            raise ValueError("The name of the environment is empty")
        return super().__new__(cls)

    def __init__(
        self,
        action_space: gymnasium.spaces.Space,
        dt: float,
        dtype: str,
        observation_space: gymnasium.spaces.Space,
        state_space: gymnasium.spaces.Space,
        seed: int,
        interdecision_time_dist="constant",
    ) -> None:
        # Mandatory attributes
        # self.state: npt.NDArray[np.floating]
        # self.observation: npt.NDArray[np.floating]
        # self.action: npt.NDArray[np.floating]

        # Space must be of type gymnasium.spaces.Box
        assert isinstance(state_space, gymnasium.spaces.Box)
        assert isinstance(action_space, gymnasium.spaces.Box)
        assert isinstance(observation_space, gymnasium.spaces.Box)

        self.state_space = state_space
        self.action_space = action_space
        self.observation_space = observation_space

        # Seeding the spaces
        self.state_space.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

        self.dt = dt

        # Set dtype
        self.dtype = dtype
        np_dtype = np.dtype(self.dtype)
        assert np_dtype.type in [np.float32, np.float64], "Only float32/64 supported"
        self.np_dtype = np_dtype

        # Check that the spaces are of the same dtype
        assert self.state_space.dtype == self.np_dtype
        assert self.action_space.dtype == self.np_dtype
        assert self.observation_space.dtype == self.np_dtype

        # Set the seed
        self.seed_env = seed

        # Set the distribution of the interdecision time
        assert interdecision_time_dist in [
            "constant",
            "uniform",
            "exponential",
        ], "The distribution of the interdecision time is not implemented"
        self.interdecision_time_dist = interdecision_time_dist

        # Set the buffer for the history
        self._list_time_points: list[float] = []  # Buffer of time points
        self._list_array_obs_history: list[
            npt.NDArray
        ] = []  # Buffer of past observations
        self._list_array_state_history: list[npt.NDArray] = []  # Buffer of past states

    def sample_interdecision_time(self) -> float:
        """Sample the interdecision time from the distribution."""
        if self.interdecision_time_dist == "constant":
            return self.dt
        elif self.interdecision_time_dist == "uniform":
            return np.random.uniform(0, 2 * self.dt)
        elif self.interdecision_time_dist == "exponential":
            return np.random.exponential(self.dt)
        else:
            raise ValueError("The distribution is not implemented")

    @abstractmethod
    def cost_function(
        self,
        time: float,
        array_control: npt.NDArray,
        array_observation: npt.NDArray,
        array_observation_next: npt.NDArray,
    ) -> float:
        """Static cost function: c(o_t, u_t, o_{t+1}).

        Here o represents the observation. The thing is that, some environments
        might not have access to the state, so the cost function is defined in terms
        of the observation.
        An example of this is when a learnt model of the observations of a dynamics
        is used. No access to the state is given, only the observations.

        Args:
            array_observation: A numpy.array, the observation of the dynamics.
            array_observation_next: A numpy.array, the next observation reached after
                applying control.
            array_control: A numpy.array, the control input to the cost function.
            time: A float, the current time for non-stationary costs.

        Returns:
            A float, the value c(x_t, u_t, x_{t+1}).

        """
        pass

    @property
    def current_t(self) -> float:
        """Return the current time."""
        return self.array_time_points[-1]

    @property
    def current_step(self) -> int:
        """Return the current step."""
        if self.array_time_points.size == 0:
            raise AttributeError("The time points are empty")

        return len(self.array_time_points) - 1

    @property
    @abstractmethod
    def array_state(self) -> npt.NDArray[np.floating]:
        pass

    # Define the setter for the state
    @array_state.setter
    @abstractmethod
    def array_state(self, array_x: npt.NDArray[np.floating]) -> None:
        pass

    @property
    @abstractmethod
    def array_observation(self) -> npt.NDArray[np.floating]:
        pass

    # No setter for the observations because they are derived from the state
    # by definition (Y = g(X))

    @property
    @abstractmethod
    def array_control(self) -> npt.NDArray[np.floating] | None:
        pass

    @property
    def array_time_points(self) -> npt.NDArray[np.floating]:
        """Return the list of time points."""
        return np.array(self._list_time_points, self.dtype)

    def translation_array_time_points(self, time: float) -> None:
        """Translate the time points."""
        self._list_time_points = [t - time for t in self._list_time_points]

    @property
    def list_array_obs_history(self) -> list[npt.NDArray]:
        """Return the list of observations."""
        return self._list_array_obs_history

    @property
    def list_array_state_history(self) -> list[npt.NDArray]:
        """Return the list of states."""
        return self._list_array_state_history

    @property
    def unwrapped(self) -> Env:
        """Return this environment in its base (most inner) form.

        Returns:
            The base instance of this environment.
        """
        return self
