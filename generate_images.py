#!/usr/bin/env python3
import cv2, gym, numpy as np, os, random, rl, tensorflow as tf, time
cv2.ocl.setUseOpenCL(False)

from lib import common
from matplotlib import pyplot as plt
from PIL import Image
from rl.common.wrappers import WarpFrame


base_dir = 'D:/Models/'

# Use hyper-parameters for breakout
run_name = 'breakout'
params = common.HYPERPARAMS[run_name]
env  = rl.common.wrappers.make_atari(params['env_name'])
env  = rl.common.wrappers.wrap_deepmind(env=env, warp_frames=False, clip_rewards=False)
n_actions = env.action_space.n

with tf.Session() as sess:
	# Load trained network
	subdir = os.path.join(run_name, '_test')
	saver = tf.train.import_meta_graph(os.path.join(base_dir, subdir, subdir, '-12487501.meta'))
	saver.restore(sess, tf.train.latest_checkpoint(os.path.join(base_dir, subdir)))
	graph = tf.get_default_graph()
	X_state = graph.get_tensor_by_name('state:0')
	dense_tense = graph.get_tensor_by_name(os.path.join('online', 'dense', 'Relu:0'))
	q_vals = graph.get_tensor_by_name(os.path.join('online', 'dense_1', 'MatMul:0'))

	# Run game
	i1 = 0
	N = 120000
	all_q = np.empty((0, env.action_space.n))
	all_dense = np.empty((0, 512))
	all_time = np.empty((0, 1))

	while True:
		obs_col = env.reset()
		i2 = np.zeros((1, 1))
		while True:
			if random.random() < 0.05:
				action = random.randint(0, n_actions-1)
			else:
				# Save colour and grayscale images
				Image.fromarray(obs_col[0:210, ...]).save(
					os.path.join(os.pardir, 'Visualisations', run_name, 'col_frames', f'{i1}.png'))
				obs_gray = np.zeros((4, 84, 84))
				for j in range(4):
					frame = np.array(obs_col)[j*210:(j+1)*210, ...]
					frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
					obs_gray[j, ...] = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
				Image.fromarray(obs_gray[0, ...].astype(np.uint8), mode='L').save(
					os.path.join(os.pardir, 'Visualisations', run_name, 'gray_frames', f'{i1}.png'))

				# Compute partial and complete forward pass
				dense = dense_tense.eval({X_state: obs_gray[None, ...]})
				q = q_vals.eval({X_state: obs_gray[None, ...]})
				action = q.argmax(axis=1)

				# Append to existing arrays
				all_q = np.concatenate((all_q, q), axis=0)
				all_dense = np.concatenate((all_dense, dense), axis=0)
				all_time = np.concatenate((all_time, i2), axis=0)

				# Count all frames a decision was made on
				i1 += 1
				if i1 % 1000 == 0:
					print(i1)
				if i1 == N:
					break
			# Count frame number within current game
			i2 += 1

			obs_col, reward, done, info = env.step(action)

			if info['ale.lives'] == 0:
				break

		if i1 == N:
			break

	# Save useful arrays
	np.savetxt(os.path.join(os.pardir, 'Visualisations', run_name, 'hidden_activations.tsv'),
		all_dense, delimiter='\t')
	np.savetxt(os.path.join(os.pardir, 'Visualisations', run_name, 'q_values.tsv'),
		all_q, delimiter='\t')
	np.savetxt(os.path.join(os.pardir, 'Visualisations', run_name, 'time_num.tsv'),
		all_time, delimiter='\t')
