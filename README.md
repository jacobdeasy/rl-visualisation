# Human-Level Control through Deep Reinforcement Learning

Tensorflow implementation of [Human-level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

This implementation contains:

1. Deep Q-network and Q-learning
2. Experience replay memory
	- to reduce the correlations between consecutive updates
3. Network for Q-learning targets are fixed for intervals
	- to reduce the correlations between target and predicted Q-values

Agent attention from my MPhil project on deep reinforcement learning!
![]('Assets/breakout_game.gif')

How the agent learnt to focus through time!
![]('Assets/breakout_training.gif')

## Requirements
- OpenAI gym
- Tensorflow
- Numpy
- Collections

and all of their dependencies.

## References
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)