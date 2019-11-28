#!/usr/bin/env python3
import gym, numpy as np, os, random, rl, tensorflow as tf, time

from lib import common
from matplotlib import pyplot as plt
from PIL import Image


base_dir = 'D:/Models/'

# Use hyper-parameters for breakout
run_name = 'breakout'
params = common.HYPERPARAMS[run_name]
env  = rl.common.wrappers.make_atari(params['env_name'])
env  = rl.common.wrappers.wrap_deepmind(env=env, clip_rewards=False)
n_actions = env.action_space.n

with tf.Session() as sess:
	# Load trained network
	subdir = run_name+'_test'
	saver = tf.train.import_meta_graph(os.path.join(base_dir, subdir, subdir+'-12487501.meta'))
	# saver = tf.train.import_meta_graph(base_dir+subdir+'/'+subdir+'-10565626.meta')
	saver.restore(sess, tf.train.latest_checkpoint(base_dir+subdir))
	graph = tf.get_default_graph()
	X_state = graph.get_tensor_by_name('state:0')
	dense_tense = graph.get_tensor_by_name(os.path.join('online', 'dense', 'BiasAdd:0'))
	q_vals = graph.get_tensor_by_name(os.path.join('online', 'dense_1', 'MatMul:0'))

	# Run game
	i = 0
	obs = env.reset()
	total_reward = 0
	won, lost, died = [], [], []
	all_q = np.empty((0, env.action_space.n))
	all_dense = np.empty((0, 512))
	while True:
		time.sleep(0.1)
		env.render()
		if random.random() < 0.05:
			q = np.empty((1, env.action_space.n))
			q[:] = np.nan
			dense = np.empty((1, 512))
			dense[:] = np.nan
			action = random.randint(0, n_actions-1)
		else:
			dense = dense_tense.eval({X_state: obs[None, ...]})
			q = q_vals.eval({X_state: obs[None, ...]})
			action = q.argmax(axis=1)
		all_q = np.concatenate((all_q, q), axis=0)
		all_dense = np.concatenate((all_dense, dense), axis=0)

		obs, reward, done, info = env.step(action)
		total_reward += reward
		if info['ale.lives'] == 0:
			break

		if reward > 0:
			won.append(i)
		elif reward < 0:
			lost.append(i)
		elif done:
			died.append(i)
			env.reset()
		i += 1

	# Plot
	mask = np.isfinite(all_q[:, 0])
	for i in range(env.action_space.n):
		plt.plot(all_q[mask, i])
	for xc in won:
		plt.axvline(x=xc, color='blue', linestyle=':')
	for xc in lost:
		plt.axvline(x=xc, color='red', linestyle=':')
	for xc in died:
		plt.axvline(x=xc, color='black')
	plt.legend(env.unwrapped.get_action_meanings())
	plt.show()
