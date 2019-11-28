import gym
import torch
import random
import collections
from torch.autograd import Variable

import numpy as np

from collections import namedtuple, deque

from .agent import BaseAgent

# One single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments.
    Every experience contains n list of Experience entries.
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        # Loop through list of environments, NB. THIS LOOP IS USED REPEATEDLY.
        for env in self.pool:
            obs = env.reset()
            # If the environment is vectorized, its output is a list of results.
            # Details are here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_state())

        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    # Select random action
                    actions[idx] = self.pool[0].action_space.sample()
                else:
                    # Add state to state_inputs
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                # Convert states to actions via agent
                states_actions, new_agent_states = self.agent(states_input, agent_states)
                for idx, action in enumerate(states_actions):
                    # Grouped actions and agent_states
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            # Group action lists
            grouped_actions = _group_list(actions, env_lens)

            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                # Take step
                if self.vectorized:
                    next_state_n, r_n, is_done_n, _ = env.step(action_n)
                else:
                    next_state, r, is_done, _ = env.step(action_n[0])
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]

                # Loop through list output from env.step
                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    # Extract current state and history, update current rewards and steps
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]
                    cur_rewards[idx] += r
                    cur_steps[idx] += 1

                    if state is not None:
                        # Add state to history
                        history.append(Experience(state=state, action=action, reward=r, done=is_done))
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        # 
                        yield tuple(history)
                    states[idx] = next_state
                    if is_done:
                        # Generate tail of history
                        while len(history) >= 1:
                            yield tuple(history)
                            history.popleft()
                        # Add current vars to total vars
                        self.total_rewards.append(cur_rewards[idx])
                        self.total_steps.append(cur_steps[idx])
                        # Reset current vars
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # Vectorized envs are reset automatically
                        states[idx] = env.reset() if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        # Clear history
                        history.clear()
                global_ofs += len(action_n)
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


def _group_list(items, lens):
    """
    Unflat the list of items by lens
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    res = []
    cur_ofs = 0
    for g_len in lens:
        res.append(items[cur_ofs:cur_ofs+g_len])
        cur_ofs += g_len
    return res


# Entries that are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    Wrapper around ExperienceSource to prevent storing full trajectory in replay buffer.
    For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.
    If we have partial trajectory at the end of episode, last_state will be None.
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        assert isinstance(gamma, float)
        # Child-class of base-class ExperienceSource
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta, vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)


class ExperienceSourceBuffer:
    """
    The same as ExperienceSource, but takes episodes from the buffer
    """
    def __init__(self, buffer, steps_count=1):
        """
        Create buffered experience source
        :param buffer: list of episodes, each is a list of Experience object
        :param steps_count: count of steps in every entry
        """
        self.update_buffer(buffer)
        self.steps_count = steps_count

    def update_buffer(self, buffer):
        self.buffer = buffer
        self.lens = list(map(len, buffer))

    def __iter__(self):
        """
        Infinitely sample episode from the buffer and then sample item offset
        """
        while True:
            episode = random.randrange(len(self.buffer))
            ofs = random.randrange(self.lens[episode] - self.steps_count - 1)
            yield self.buffer[episode][ofs:ofs+self.steps_count]


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)
