

def use_proper_time_limits(env):
    return hasattr(env, 'get_max_episode_steps') and env.get_max_episode_steps() is not None

def get_obs_at_index(obs, i):
    if isinstance(obs, dict):
        return {k: obs[k][i] for k in obs.keys()}
    else:
        return obs[i]
    
def set_obs_at_index(obs, obs_, i):
    if isinstance(obs, dict):
        for k in obs.keys():
            obs[k][i] = obs_[k].squeeze(0)
    else:
        obs[i] = obs_[0].squeeze(0)

def is_discrete_actions(env, adversary=False):
    if adversary:
        return env.adversary_action_space.__class__.__name__ == 'Discrete' # todo: remove if not allowing paired
    else:
        return env.action_space.__class__.__name__ == 'Discrete'
    
def make_plr_args(env, config):
    return dict(
        seeds=[],
        obs_space=env.observation_space,
        action_space=env.action_space,
        num_actors=config['num_actors'],
        strategy=config['level_replay_strategy'],
        replay_schedule=config['level_replay_schedule'],
        score_transform=config['level_replay_score_transform'],
        temperature=config['level_replay_temperature'],
        eps=config['level_replay_eps'],
        rho=config['level_replay_rho'],
        replay_prob=config['level_replay_prob'],
        alpha=config['level_replay_alpha'],
        staleness_coef=config['staleness_coef'],
        staleness_transform=config['staleness_transform'],
        staleness_temperature=config['staleness_temperature'],
        sample_full_distribution=config['train_full_distribution'],
        seed_buffer_size=config['level_replay_seed_buffer_size'],
        seed_buffer_priority=config['level_replay_seed_buffer_priority'],
        gamma=config['gamma']
    )