import gym
import random
import collections

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
    def __init__(self, env, agent, steps_count=2, steps_delta=1):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        self.env = env
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []

    def __iter__(self):
        # Initialise reward, state and trajectory history
        state = self.env.reset()
        history = deque(maxlen=self.steps_count)
        cur_rewards = 0.0
        agent_state = self.agent.initial_state()

        idx = 0
        while True:
            # Convert state to action via agent
            action, new_agent_state = self.agent([state], agent_state)

            # Take step
            next_state, r, is_done, _ = self.env.step(action[0])
            cur_rewards += r

            # Add state to history
            history.append(Experience(state=state, action=action[0], reward=r, done=is_done))
            if len(history) == self.steps_count and idx % self.steps_delta == 0:
                yield tuple(history)
            state = next_state

            if is_done:
                # Generate tail of history
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.total_rewards.append(cur_rewards)
                cur_rewards = 0.0
                # Reset env
                state = self.env.reset()
                agent_state = self.agent.initial_state()
                # Clear history #### uneccessary
                history.clear()
            idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
        return r


# Entries that are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    Wrapper around ExperienceSource to prevent storing full trajectory in replay buffer.
    For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.
    If there is a partial trajectory at the end of episode, last_state will be None.
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1):
        assert isinstance(gamma, float)
        # Child-class of base-class ExperienceSource
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta)
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


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

    def populate(self, num_samples):
        for _ in range(num_samples):
            entry = next(self.experience_source_iter)
            self._add(entry)
