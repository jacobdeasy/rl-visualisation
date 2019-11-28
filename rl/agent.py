"""
Agent is something which converts states into actions.
"""
import copy
import tensorflow as tf


class BaseAgent:
    """
    Abstract Agent Interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def  __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)
        raise NotImplementedError


class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values from the
    observations and converts them into the actions using action_selector
    """
    def __init__(self, X_state, net_q, action_selector):
        self.X_state = X_state
        self.net_q = net_q
        self.action_selector = action_selector
        self.c = 0

    def __call__(self, state, agent_states):
        # Get Q values for states by forward pass through model on GPU
        q_values = self.net_q.eval({self.X_state: state})
        # Select action based on Q values and action_selector
        action = self.action_selector(q_values)
        return action, agent_states
