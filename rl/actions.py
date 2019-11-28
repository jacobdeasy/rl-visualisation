import numpy as np


class ActionSelector:
	"""
	Abstract class which converts scores to the actions
	"""
	def __call__(self, scores):
		raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
	"""
	Selects actions using argmax
	"""
	def __call__(self, scores):
		assert isinstance(scores, np.ndarray)
		return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
	"""
	Select actions randomly at rate epsilon
	"""
	def __init__(self, epsilon=0.05, selector=None):
		self.epsilon  = epsilon
		self.selector = selector if selector is not None else ArgmaxActionSelector()

	def __call__(self, scores):
		assert isinstance(scores, np.ndarray)
		# scores is a batch_size x n_actions array
		batch_size, n_actions = scores.shape
		# Select all actions by argmax
		actions = self.selector(scores)
		# Create mask of random action indices
		mask = np.random.random(size=batch_size) < self.epsilon
		# Randomly choose actions
		rand_actions = np.random.choice(n_actions, sum(mask))
		# Overlay random actions onto actions
		actions[mask] = rand_actions
		return actions
