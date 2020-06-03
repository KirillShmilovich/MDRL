from simulation import LJSimualtion
from simtk import unit as u
import numpy as np
from gym import spaces
import gym


class SimEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 min_temp=1 * u.kelvin,
                 max_temp=500 * u.kelvin,
                 delta=5 * u.kelvin,
                 **kwargs):
        super(SimEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.delta = delta
        self.action_space = spaces.Box(low=-1.,
                                       high=1.,
                                       shape=(1, ),
                                       dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0.,
                                            high=1.,
                                            shape=(74, ),
                                            dtype=np.float32)

        self.reset()

    def step(self, action):
        n_steps = 500
        k = 20.

        deltaT = self.delta * action
        new_temp = self._temp_clamp(self.sim.temperature + deltaT)
        self.sim.changeTemp(new_temp)

        self.sim.sim(n_steps)

        observation = self._state()

        n_clusters = self.sim.n_clusters()
        reward = -(1 - n_clusters / k)**2

        done = 0
        info = {'n_clusters': n_clusters}

        return observation, reward, done, info

    def _temp_clamp(self, x):
        return max(self.min_temp, min(self.max_temp, x))

    def _scale(self, x):
        z = (x - self.min_temp) / (self.max_temp - self.min_temp)
        return z

    def _state(self):
        state = np.array([self._scale(self.sim.temperature)])
        state = np.concatenate((state, self.sim.graphlet_features()))
        return state

    def reset(self):
        temp = np.random.uniform(
            low=self.min_temp._value,
            high=self.max_temp._value) * self.min_temp.unit
        self.sim = LJSimualtion(temperature=temp)
        observation = self._state()
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == 'main':
    from stable_baselines3.common.env_checker import check_env
    env = SimEnv()
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)
