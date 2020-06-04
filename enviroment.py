from simulation import LJSimualtion, ADPSimualtion
from simtk import unit as u
import numpy as np
from gym import spaces
import gym


class LJSimEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        min_temp=1 * u.kelvin,
        max_temp=500 * u.kelvin,
        delta=5 * u.kelvin,
        episode_duration=1000,
        report=False,
        **kwargs
    ):
        super(LJSimEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.delta = delta
        self.episode_duration = episode_duration
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4951,), dtype=np.float32
        )

        self.reset(report=report)

    def step(self, action):
        n_steps = 500
        k = 20.0

        # Change temp
        deltaT = self.delta * action[0]
        new_temp = self._temp_clamp(self.sim.temperature + deltaT)
        self.sim.changeTemp(new_temp)

        # Run sim and update episode step
        self.sim.sim(n_steps)
        self.episode_step += 1

        # Get the state
        observation = self._state()

        # Calculate reward
        n_clusters = self.sim.n_clusters()
        reward = -((1 - n_clusters / k) ** 2)
        info = {}
        self.info_dict = {
            "n_clusters": n_clusters,
            "temperature": self.sim.temperature,
            "reward": reward,
        }

        # Check if done
        if self.episode_step >= self.episode_duration:
            done = 1
        else:
            done = 0

        return observation, reward, done, info

    def _temp_clamp(self, x):
        return max(self.min_temp, min(self.max_temp, x))

    def _scale(self, x):
        z = (x - self.min_temp) / (self.max_temp - self.min_temp)
        return z

    def _state(self):
        state = np.array([self._scale(self.sim.temperature)])
        # state = np.concatenate((state, self.sim.graphlet_features()))
        state = np.concatenate((state, self.sim.pairwise_features()))
        return state

    def reset(self, report=False):
        temp = (
            np.random.uniform(low=self.min_temp._value, high=self.max_temp._value)
            * self.min_temp.unit
        )
        self.sim = LJSimualtion(report=report, temperature=temp)
        observation = self._state()
        self.episode_step = 0
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class ADPSimEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        min_temp=1 * u.kelvin,
        max_temp=500 * u.kelvin,
        delta=5 * u.kelvin,
        episode_duration=1000,
        report=False,
        **kwargs
    ):
        super(ADPSimEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.delta = delta
        self.episode_duration = episode_duration
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(232,), dtype=np.float32
        )

        self.reset(report=report)

    def step(self, action):
        n_steps = 500

        # Change temp
        deltaT = self.delta * action[0]
        new_temp = self._temp_clamp(self.sim.temperature + deltaT)
        self.sim.changeTemp(new_temp)

        # Run sim and update episode step
        self.sim.sim(n_steps)
        self.episode_step += 1

        # Get the state
        observation = self._state()

        # Calculate reward
        rg = self.sim.rg()
        reward = -1 * rg
        info = {}
        self.info_dict = {
            "rg": rg,
            "temperature": self.sim.temperature,
            "reward": reward,
        }

        # Check if done
        if self.episode_step >= self.episode_duration:
            done = 1
        else:
            done = 0

        return observation, reward, done, info

    def _temp_clamp(self, x):
        return max(self.min_temp, min(self.max_temp, x))

    def _scale(self, x):
        z = (x - self.min_temp) / (self.max_temp - self.min_temp)
        return z

    def _state(self):
        state = np.array([self._scale(self.sim.temperature)])
        # state = np.concatenate((state, self.sim.graphlet_features()))
        state = np.concatenate((state, self.sim.pairwise_features()))
        return state

    def reset(self, report=False):
        temp = (
            np.random.uniform(low=self.min_temp._value, high=self.max_temp._value)
            * self.min_temp.unit
        )
        self.sim = ADPSimualtion(report=report, temperature=temp)
        observation = self._state()
        self.episode_step = 0
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass


if __name__ == "main":
    from stable_baselines3.common.env_checker import check_env

    env = ADPSimEnv()
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)
