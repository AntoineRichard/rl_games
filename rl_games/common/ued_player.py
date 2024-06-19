import time
import gym
import numpy as np
import torch
import copy
from rl_games.common import vecenv
from rl_games.common import env_configurations
from rl_games.algos_torch import model_builder
from rl_games.common.player import BasePlayer
from rl_games.rl_games.common.ued_utils.storage import RolloutStorage

from collections import deque, defaultdict
from rl_games.rl_games.common.ued_utils.level_store import LevelStore
from rl_games.rl_games.common.ued_utils.level_sampler import LevelSampler
from rl_games.rl_games.common.ued_utils import is_discrete_actions, get_obs_at_index, set_obs_at_index, make_plr_args

# currently supported UED algorithms
UED_ALGOS = ['paired', 'dr', 'plr', 'accel'] # assume domain randomization for PLR (no adversarial agent)


class BaseUEDPlayer(BasePlayer):

    def __init__(self, params):
        super().__init__(params)
        ued_algo = params['ued_algo']
        assert self.ued_algo in UED_ALGOS

        self.num_processes = self.config['num_actors']
        self.agent_rollout_steps = self.env._max_episode_steps # todo: check if correct gym.Env param

        self.is_discrete_actions = is_discrete_actions(self.env)
        self.is_discrete_adversary_env_actions = is_discrete_actions(self.env, adversary=True)

        self.is_dr = ued_algo == 'domain_randomization'
        self.is_paired = ued_algo == 'paired'
        self.use_plr = ued_algo == 'plr' or ued_algo == 'accel'
        self.is_accel = ued_algo == 'accel'

        adversary_agent, adversary_env = None, None
        if self.is_paired: # todo: add make_agent function because require custom logic for each environment
            adversary_agent = make_agent(name='adversary_agent', env=self.env, args=self.config)
            adversary_env = make_agent(name='adversary_env', env=self.env, args=self.config)

        self.agents = {
            'agent': self, # todo: check
            'adversary_agent': adversary_agent,
            'adversary_env': adversary_env,
        }

        self.adversary_env_rollout_steps = self.env.adversary_observation_space['time_step'].high[0]

        self.level_store = None
        self.level_samplers = {}
        self.current_level_seeds = None
        self.weighted_num_edits = 0
        self.latest_env_stats = defaultdict(float)

        self.plr_args = None
        if self.use_plr:
            self.plr_args = make_plr_args(self.env, self.config) # todo: check

        if self.plr_args:
            if self.is_paired:
                if not self.config['protagonist_plr'] and not self.config['antagonist_plr']:
                    self.level_samplers.update({
                        'agent': LevelSampler(**self.plr_args),
                        'adversary_agent': LevelSampler(**self.plr_args)
                    })
                elif self.config['protagonist_plr']:
                    self.level_samplers['agent'] = LevelSampler(**self.plr_args)
                elif self.config['antagonist_plr']:
                    self.level_samplers['adversary_agent'] = LevelSampler(**self.plr_args)
            else:
                self.level_samplers['agent'] = LevelSampler(**self.plr_args)

            if self.env.use_byte_encoding:
                example = self.env.get_encodings()[0] # todo: how to batch retrieve property from vecenv
                data_info = {
                    'numpy': True,
                    'dtype': example.dtype,
                    'shape': example.shape
                }
                self.level_store = LevelStore(data_info=data_info)
            else:
                self.level_store = LevelStore()

            self.current_level_seeds = [-1 for i in range(self.num_processes)]

            self._default_level_sampler = self.all_level_samplers[0]

            self.use_editor = self.is_accel
            self.edit_prob = self.config['level_editor_prob']
            self.base_levels = self.config['base_levels']
        else:
            self.use_editor = False
            self.edit_prob = 0
            self.base_levels = None

        self.reset_runner() 

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if 'obs' in obs:
                obs = obs['obs']
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert (obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(
                weights['running_mean_std'])

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_deterministic=False):
        raise NotImplementedError('step')

    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size(
            )[2]), dtype=torch.float32).to(self.device) for s in rnn_states]


    # ============================== UED Functions ==============================

    def reset_runner(self):
        self.num_updates = 0
        self.total_num_edits = 0
        self.total_episodes_collected = 0
        self.total_seeds_collected = 0
        self.student_grad_updates = 0
        self.sampled_level_info = None

        max_return_queue_size = 10
        self.agent_returns = deque(maxlen=max_return_queue_size)
        self.adversary_agent_returns = deque(maxlen=max_return_queue_size)

    @property
    def all_level_samplers(self):
        if len(self.level_samplers) == 0:
            return []

        return list(filter(lambda x: x is not None, [v for _, v in self.level_samplers.items()]))
    
    def _sample_replay_decision(self):
        return self._default_level_sampler.sample_replay_decision()
    
    def _get_batched_value_loss(self, agent, clipped=True, batched=True):
        batched_value_loss = agent.storage.get_batched_value_loss(
            signed=False,
            positive_only=False,
            clipped=clipped,
            batched=batched)

        return batched_value_loss

    def _get_rollout_return_stats(self, rollout_returns):
        mean_return = torch.zeros(self.num_processes, 1)
        max_return = torch.zeros(self.num_processes, 1)
        for b, returns in enumerate(rollout_returns):
            if len(returns) > 0:
                mean_return[b] = float(np.mean(returns))
                max_return[b] = float(np.max(returns))

        stats = {
            'mean_return': mean_return,
            'max_return': max_return,
            'returns': rollout_returns
        }

        return stats
    
    def _get_active_levels(self):
        assert self.use_plr, 'Only call _get_active_levels when using PLR/ACCEL.'

        if self.use_byte_encoding:
            return [x.tobytes() for x in self.env.get_encodings()] # todo: mass call
        else:
            return self.env.get_level() # todo: mass call
    
    def _update_level_samplers_with_external_unseen_sample(self, seeds, solvable=None):
        level_samplers = self.all_level_samplers

        # if self.args.reject_unsolvable_seeds:
        #     solvable = np.array(solvable, dtype=np.bool)
        #     seeds = np.array(seeds, dtype=np.int)[solvable]
        #     solvable = solvable[solvable]

        for level_sampler in level_samplers:
            level_sampler.observe_external_unseen_sample(seeds, solvable)

    def _update_plr_with_current_unseen_levels(self, parent_seeds=None):
        args = self.args
        levels = self._get_active_levels()
        self.current_level_seeds = \
            self.level_store.insert(levels, parent_seeds=parent_seeds)
        # if args.log_plr_buffer_stats or args.reject_unsolvable_seeds:
        #     passable = self.venv.get_passable()
        # else:
        passable = None
        self._update_level_samplers_with_external_unseen_sample(
            self.current_level_seeds, solvable=passable)
        
    def _should_edit_level(self):
        if self.use_editor:
            return np.random.rand() < self.edit_prob
        else:
            return False
        
    def _reconcile_level_store_and_samplers(self):
        all_replay_seeds = set()
        for level_sampler in self.all_level_samplers:
            all_replay_seeds.update([x for x in level_sampler.seeds if x >= 0])
        self.level_store.reconcile_seeds(all_replay_seeds)

    def _get_weighted_num_edits(self):
        level_sampler = self.all_level_samplers[0]
        seed_num_edits = np.zeros(level_sampler.seed_buffer_size)
        for idx, value in enumerate(self.level_store.seed2parent.values()):
            seed_num_edits[idx] = len(value)
        weighted_num_edits = np.dot(level_sampler.sample_weights(), seed_num_edits)
        return weighted_num_edits
    
    def _compute_env_return(self, agent_info, adversary_agent_info):
        args = self.args
        if args.ued_algo == 'paired':
            env_return = torch.max(adversary_agent_info['max_return'] - agent_info['mean_return'], \
                torch.zeros_like(agent_info['mean_return']))

        elif args.ued_algo == 'flexible_paired':
            env_return = torch.zeros_like(agent_info['max_return'], dtype=torch.float, device=self.device)
            adversary_agent_max_idx = adversary_agent_info['max_return'] > agent_info['max_return']
            agent_max_idx = ~adversary_agent_max_idx

            env_return[adversary_agent_max_idx] = \
                adversary_agent_info['max_return'][adversary_agent_max_idx]
            env_return[agent_max_idx] = agent_info['max_return'][agent_max_idx]

            env_mean_return = torch.zeros_like(env_return, dtype=torch.float)
            env_mean_return[adversary_agent_max_idx] = \
                agent_info['mean_return'][adversary_agent_max_idx]
            env_mean_return[agent_max_idx] = \
                adversary_agent_info['mean_return'][agent_max_idx]

            env_return = torch.max(env_return - env_mean_return, torch.zeros_like(env_return))

        elif args.ued_algo == 'minimax':
            env_return = -agent_info['max_return']

        elif args.ued_algo == 'perm':
            env_params = self.venv.get_obs()
            param_max = self.venv.param_info['param_max']
            param_min = self.venv.param_info['param_min']
            params = []
            for param in env_params:
                params_normed = (param - param_min)/(param_max - param_min)
                params.append(params_normed)
            params_tensor = torch.tensor(np.array(params), dtype = torch.float, device = self.device)
            normed_agent_rewards = torch.tensor(self.perm.scaler.transform(agent_info['mean_return']), dtype = torch.float, device = self.device)
            env_rewards, ability = self.perm.estimate_rewards(params_tensor, normed_agent_rewards)
            self.ued_venv.update_ability(ability.detach().cpu().numpy())
            env_return = torch.mean(env_rewards.detach())
        else:
            env_return = torch.zeros_like(agent_info['mean_return'])

        if args.adv_normalize_returns:
            self.env_return_rms.update(env_return.flatten().cpu().numpy())
            env_return /= np.sqrt(self.env_return_rms.var + 1e-8)

        if args.adv_clip_reward is not None:
            clip_max_abs = args.adv_clip_reward
            env_return = env_return.clamp(-clip_max_abs, clip_max_abs)

        return env_return

    def ued_rollout(self,
                    agent,
                    num_steps,
                    update=False,
                    is_env=False,
                    level_replay=False,
                    level_sampler=None,
                    update_level_sampler=False,
                    discard_grad=False,
                    edit_level=False,
                    num_edits=0,
                    fixed_seeds=None):
        if is_env:
            if edit_level: # Get mutated levels
                levels = [self.level_store.get_level(seed) for seed in fixed_seeds]
                self.env.reset_to_level_batch(levels) # todo: mass call
                self.env.mutate_level(num_edits=num_edits) # todo: mass call
                self._update_plr_with_current_unseen_levels(parent_seeds=fixed_seeds)
                return
            if level_replay: # Get replay levels
                self.current_level_seeds = [level_sampler.sample_replay_level() for _ in range(self.num_processes)]
                levels = [self.level_store.get_level(seed) for seed in self.current_level_seeds]
                self.env.reset_to_level_batch(levels) # todo: mass call
                return self.current_level_seeds
            elif self.use_plr:
                obs = self.env.reset_random() # todo: mass call
                self._update_plr_with_current_unseen_levels(parent_seeds=fixed_seeds)
                self.total_seeds_collected += self.num_processes
                return
            else:
                obs = self.env.reset() # todo: mass call
                self.total_seeds_collected += self.num_processes
        else:
            obs = self.env.reset_agent() # todo: mass call

        agent.storage.copy_obs_to_index(obs,0)

        rollout_info = {}
        rollout_returns = [[] for _ in range(self.num_processes)]

        for step in range(num_steps):
            if self.render: # todo: check whether there is flag for environment rendering
                self.env.render_to_screen()
            # Sample actions
            with torch.no_grad():
                obs_id = agent.storage.get_obs(step)
                value, action, action_log_dist, recurrent_hidden_states = agent.act( # todo: replace with get_action for rlgames agents
                    obs_id, agent.storage.get_recurrent_hidden_state(step), agent.storage.masks[step])
                if is_env: # is paired
                    if self.is_discrete_adversary_env_actions:
                        action_log_prob = action_log_dist.gather(-1, action)
                    else:
                        action_log_prob = action_log_dist
                else:
                    if self.is_discrete_actions:
                        action_log_prob = action_log_dist.gather(-1, action)
                    else:
                        action_log_prob = action_log_dist

            # Observe reward and next obs
            reset_random = self.is_dr and not self.use_plr
            _action = agent.process_action(action.cpu())

            if is_env:
                obs, reward, done, infos = self.env.step_adversary(_action) # todo: mass call
            else:
                obs, reward, done, infos = self.env.step_env(_action, reset_random=reset_random) # todo: replace with self.step_env()
                if self.config['clip_reward']:
                    reward = torch.clamp(reward, -self.config['clip_reward'], self.config['clip_reward'])

            if not is_env and step >= num_steps - 1:
                # Handle early termination due to cliffhanger rollout
                if agent.storage.use_proper_time_limits:
                    for i, done_ in enumerate(done):
                        if not done_:
                            infos[i]['cliffhanger'] = True
                            infos[i]['truncated'] = True
                            infos[i]['truncated_obs'] = get_obs_at_index(obs, i)

                done = np.ones_like(done, dtype=np.float)

            if level_sampler and level_replay:
                next_level_seeds = [s for s in self.current_level_seeds]

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    rollout_returns[i].append(info['episode']['r'])

                    if reset_random:
                        self.total_seeds_collected += 1

                    if not is_env:
                        self.total_episodes_collected += 1

                        # Handle early termination
                        if agent.storage.use_proper_time_limits:
                            if 'truncated_obs' in info.keys():
                                truncated_obs = info['truncated_obs']
                                agent.storage.insert_truncated_obs(truncated_obs, index=i)

                        # If using PLR, sample next level
                        if level_sampler and level_replay:
                            level_seed = level_sampler.sample_replay_level()
                            level = self.level_store.get_level(level_seed)
                            obs_i = self.venv.reset_to_level(level, i)
                            set_obs_at_index(obs, obs_i, i)
                            next_level_seeds[i] = level_seed
                            rollout_info['solved_idx'][i] = True

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'truncated' in info.keys() else [1.0] # todo: check RANS gym version, return truncated or store in info?
                 for info in infos])

            # Need to store level seeds alongside non-env agent steps
            current_level_seeds = None
            if (not is_env) and level_sampler:
                current_level_seeds = torch.tensor(self.current_level_seeds, dtype=torch.int).view(-1, 1)

            agent.storage.insert(
                obs, recurrent_hidden_states,
                action, action_log_prob, action_log_dist,
                value, reward, masks, bad_masks,
                level_seeds=current_level_seeds)

            if level_sampler and level_replay:
                self.current_level_seeds = next_level_seeds

        # Add generated env to level store (as a constructive string representation)
        if is_env and self.use_plr and not level_replay:
            self._update_plr_with_current_unseen_levels()

        rollout_info.update(self._get_rollout_return_stats(rollout_returns))

        # Update non-env agent if required
        if not is_env and update:
            with torch.no_grad():
                obs_id = agent.storage.get_obs(-1)
                next_value = agent.get_value(
                    obs_id, agent.storage.get_recurrent_hidden_state(-1),
                    agent.storage.masks[-1]).detach()

            agent.storage.compute_returns(
                next_value, self.config['use_gae'], self.config['gamma'], self.config['gae_lambda']) # todo: double check

            # Compute batched value loss if using value_l1-maximizing adversary
            if self.requires_batched_vloss:
                # Don't clip value loss reward if env adversary normalizes returns
                clipped = not self.config['adv_use_popart'] and not self.config['adv_normalize_returns'] # todo: check if allow reward normalization using RunningMeanStd
                batched_value_loss = self._get_batched_value_loss(
                    agent, clipped=clipped, batched=True)
                rollout_info.update({'batched_value_loss': batched_value_loss})

            # Update level sampler and remove any ejected seeds from level store
            if level_sampler and update_level_sampler:
                level_sampler.update_with_rollouts(agent.storage)

            value_loss, action_loss, dist_entropy, info = agent.update(discard_grad=discard_grad)

            if level_sampler and update_level_sampler:
                level_sampler.after_update()

            rollout_info.update({
                'value_loss': value_loss,
                'action_loss': action_loss,
                'dist_entropy': dist_entropy,
                'update_info': info,
            })

        return rollout_info
    
    def run2(self): # todo: reconcile original run with this run function
        args = self.config

        adversary_env = self.agents['adversary_env']
        agent = self.agents['agent']
        adversary_agent = self.agents['adversary_agent']

        level_replay = False
        if self.use_plr:
            level_replay = self._sample_replay_decision()

        # Discard student gradients if not level replay (sampling new levels)
        student_discard_grad = False
        no_exploratory_grad_updates = \
            vars(args).get('no_exploratory_grad_updates', False)
        if self.use_plr and (not level_replay) and no_exploratory_grad_updates:
            student_discard_grad = True

        if self.is_training and not student_discard_grad:
            self.student_grad_updates += 1

        # Generate a batch of adversarial environments
        env_info = self.ued_rollout(
            agent=adversary_env,
            num_steps=self.adversary_env_rollout_steps,
            update=False,
            is_env=True,
            level_replay=level_replay,
            level_sampler=self._get_level_sampler('agent')[0],
            update_level_sampler=False)

        # Run agent episodes
        level_sampler, is_updateable = self._get_level_sampler('agent')
        agent_info = self.ued_rollout(
            agent=agent,
            num_steps=self.agent_rollout_steps,
            update=self.is_training,
            level_replay=level_replay,
            level_sampler=level_sampler,
            update_level_sampler=is_updateable,
            discard_grad=student_discard_grad)

        # Use a separate PLR curriculum for the antagonist
        if level_replay and self.is_paired and (args.protagonist_plr == args.antagonist_plr):
            self.ued_rollout(
                agent=adversary_env,
                num_steps=self.adversary_env_rollout_steps,
                update=False,
                is_env=True,
                level_replay=level_replay,
                level_sampler=self._get_level_sampler('adversary_agent')[0],
                update_level_sampler=False)

        adversary_agent_info = defaultdict(float)
        if self.is_paired:
            # Run adversary agent episodes
            level_sampler, is_updateable = self._get_level_sampler('adversary_agent')
            adversary_agent_info = self.ued_rollout(
                agent=adversary_agent,
                num_steps=self.agent_rollout_steps,
                update=self.is_training,
                level_replay=level_replay,
                level_sampler=level_sampler,
                update_level_sampler=is_updateable,
                discard_grad=student_discard_grad)

        # Sample whether the decision to edit levels
        edit_level = self._should_edit_level() and level_replay

        if level_replay:
            sampled_level_info = {
                'level_replay': True,
                'num_edits': [len(self.level_store.seed2parent[x])+1 for x in env_info],
            }
        else:
            sampled_level_info = {
                'level_replay': False,
                'num_edits': [0 for _ in range(self.num_processes)]
            }

        # ==== This part performs ACCEL ====
        # If editing, mutate levels just replayed by PLR
        if level_replay and edit_level:
            # Choose base levels for mutation
            if self.base_levels == 'batch':
                fixed_seeds = env_info
            elif self.base_levels == 'easy':
                if self.num_processes >= 4:
                    # take top 4
                    easy = list(np.argsort((agent_info['mean_return'].detach().cpu().numpy() - agent_info['batched_value_loss'].detach().cpu().numpy()))[:4])
                    fixed_seeds = [env_info[x.item()] for x in easy] * int(self.num_processes/4)
                else:
                    # take top 1
                    easy = np.argmax((agent_info['mean_return'].detach().cpu().numpy() - agent_info['batched_value_loss'].detach().cpu().numpy()))
                    fixed_seeds = [env_info[easy]] * self.num_processes

            level_sampler, is_updateable = self._get_level_sampler('agent')

            # Edit selected levels
            self.ued_rollout(
                agent=None,
                num_steps=None,
                is_env=True,
                edit_level=True,
                num_edits=args.num_edits,
                fixed_seeds=fixed_seeds)

            self.total_num_edits += 1
            sampled_level_info['num_edits'] = [x+1 for x in sampled_level_info['num_edits']]

            # Evaluate edited levels
            agent_info_edited_level = self.ued_rollout(
                agent=agent,
                num_steps=self.agent_rollout_steps,
                update=self.is_training,
                level_replay=False,
                level_sampler=level_sampler,
                update_level_sampler=is_updateable,
                discard_grad=True)
        # ==== ACCEL end ====

        if self.use_plr:
            self._reconcile_level_store_and_samplers()
            if self.use_editor:
                self.weighted_num_edits = self._get_weighted_num_edits()

        # Update adversary agent final return
        env_return = self._compute_env_return(agent_info, adversary_agent_info)

        adversary_env_info = defaultdict(float)
        if self.is_training and self.is_training_env:
            with torch.no_grad():
                obs_id = adversary_env.storage.get_obs(-1)
                next_value = adversary_env.get_value(
                    obs_id, adversary_env.storage.get_recurrent_hidden_state(-1),
                    adversary_env.storage.masks[-1]).detach()
            adversary_env.storage.replace_final_return(env_return)
            adversary_env.storage.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
            env_value_loss, env_action_loss, env_dist_entropy, info = adversary_env.update()
            adversary_env_info.update({
                'action_loss': env_action_loss,
                'value_loss': env_value_loss,
                'dist_entropy': env_dist_entropy,
                'update_info': info
            })

        if self.is_training:
            self.num_updates += 1

        # === LOGGING ===
        # Only update env-related stats when run generates new envs (not level replay)
        log_replay_complexity = level_replay and args.log_replay_complexity
        if (not level_replay) or log_replay_complexity:
            stats = self._get_env_stats(agent_info, adversary_agent_info,
                log_replay_complexity=log_replay_complexity)
            stats.update({
                'mean_env_return': env_return.mean().item(),
                'adversary_env_pg_loss': adversary_env_info['action_loss'],
                'adversary_env_value_loss': adversary_env_info['value_loss'],
                'adversary_env_dist_entropy': adversary_env_info['dist_entropy'],
            })
            if args.use_plr:
                self.latest_env_stats.update(stats) # Log latest UED curriculum stats instead of PLR env stats
        else:
            stats = self.latest_env_stats.copy()

        # Log PLR buffer stats
        if args.use_plr and args.log_plr_buffer_stats:
            stats.update(self._get_plr_buffer_stats())

        [self.agent_returns.append(r) for b in agent_info['returns'] for r in reversed(b)]
        mean_agent_return = 0
        if len(self.agent_returns) > 0:
            mean_agent_return = np.mean(self.agent_returns)

        mean_adversary_agent_return = 0
        if self.is_paired:
            [self.adversary_agent_returns.append(r) for b in adversary_agent_info['returns'] for r in reversed(b)]
            if len(self.adversary_agent_returns) > 0:
                mean_adversary_agent_return = np.mean(self.adversary_agent_returns)

        self.sampled_level_info = sampled_level_info

        stats.update({
            'steps': (self.num_updates + self.total_num_edits) * self.num_processes * self.agent_rollout_steps,
            'total_episodes': self.total_episodes_collected,
            'total_seeds': self.total_seeds_collected,
            'total_student_grad_updates': self.student_grad_updates,

            'mean_agent_return': mean_agent_return,
            'agent_value_loss': agent_info['value_loss'],
            'agent_pg_loss': agent_info['action_loss'],
            'agent_dist_entropy': agent_info['dist_entropy'],

            'mean_adversary_agent_return': mean_adversary_agent_return,
            'adversary_value_loss': adversary_agent_info['value_loss'],
            'adversary_pg_loss': adversary_agent_info['action_loss'],
            'adversary_dist_entropy': adversary_agent_info['dist_entropy'],
        })

        if args.log_grad_norm: # todo: check if we need to log this
            agent_grad_norm = np.mean(agent_info['update_info']['grad_norms'])
            adversary_grad_norm = 0
            adversary_env_grad_norm = 0
            if self.is_paired:
                adversary_grad_norm = np.mean(adversary_agent_info['update_info']['grad_norms'])
            if self.is_training_env:
                adversary_env_grad_norm = np.mean(adversary_env_info['update_info']['grad_norms'])
            stats.update({
                'agent_grad_norm': agent_grad_norm,
                'adversary_grad_norm': adversary_grad_norm,
                'adversary_env_grad_norm': adversary_env_grad_norm
            })

        return stats
    
    # ============================== UED Functions ==============================

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                          all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards/done_count
                        cur_steps_done = cur_steps/done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.4} steps: {cur_steps_done:.4} w: {game_res}')
                        else:
                            print(f'reward: {cur_rewards_done:.4} steps: {cur_steps_done:.4f}')

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if 'obs' in obses:
                obses = obses['obs']
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if 'observation' in obses:
                first_key = 'observation'
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size
