from rl_games.common.ivecenv import IVecEnv
import gym
import numpy as np
import torch.utils.dlpack as tpack

class Envpool(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import envpool

        self.batch_size = num_actors
        env_name=kwargs.pop('env_name')
        self.has_lives = kwargs.pop('has_lives', False)
        self.env = envpool.make( env_name,
                                 env_type=kwargs.pop('env_type', 'gym'),
                                 num_envs=num_actors,
                                 batch_size=self.batch_size,
                                 **kwargs
                                )

        self.observation_space = self.env.observation_space
        self.ids = np.arange(0, num_actors)
        self.action_space = self.env.action_space
        self.scores = np.zeros(num_actors)
        self.returned_scores = np.zeros(num_actors)

    def _set_scores(self, infos, dones):
        # thanks to cleanrl: https://github.com/vwxyzjn/cleanrl/blob/3d20d11f45a5f1d764934e9851b816d0b03d2d10/cleanrl/ppo_atari_envpool.py#L111
        if 'reward' not in infos:
            return
        self.scores += infos["reward"]
        self.returned_scores[:] = self.scores
        infos["scores"] = self.returned_scores

        if self.has_lives:
            all_lives_exhausted = infos["lives"] == 0
            self.scores *= 1 - all_lives_exhausted
        else:
            # removing lives otherwise default observer will use them
            if 'lives' in infos:
                del infos['lives']
            self.scores *= 1 - dones

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action , self.ids)
        info['time_outs'] = info['TimeLimit.truncated']
        self._set_scores(info, is_done)
        return next_obs, reward, is_done, info

    def reset(self):
        obs = self.env.reset(self.ids)
        return obs

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info


def create_envpool(**kwargs):
    return Envpool("", kwargs.pop('num_actors', 16), **kwargs)