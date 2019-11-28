#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import time

from PIL import Image
from sys import platform
from utils import Slicer
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom
from skimage import color


class PostProcessing:
	"""Post-processing class."""
	def __init__(self, game, graph, X_state):
		super(PostProcessing, self).__init__()
		self.game = game
		self.graph = graph
		self.X_state = X_state

	def get_single_stack(self, img_idx, plot=False):
		frames = []
		for i in range(4):
			idx = img_idx + i - 3
			path = 'C:/Users/Owner/Documents/Project/Visualisations/'+self.game+'/gray_frames/%d.png' % idx
			frames.append(np.array(Image.open(path)))
		# BE VERY CAREFUL ABOUT THE ORDER OF STACKING HERE
		frames = np.stack(frames, axis=0)[None, ...]

		if plot:
			fig = plt.figure()
			for i in range(4):
				plt.subplot(221+i)
				plt.imshow(frames[0, i, ...], cmap=plt.cm.gray)
			fig.suptitle('Input Frames')
		return frames

	def view_conv_filter(self, conv, idx=0):
		n_out = conv.shape[3]
		for i in range(4):
			ax = plt.subplot(2, 2, i+1)
			ax.imshow(conv[..., i, idx], cmap=plt.cm.gray)
			ax.set_xticks([])
			ax.set_yticks([])
			tmp = 3-i
			ax.set_title('t-%d' % tmp)
			ax.set_aspect('auto')
		plt.subplots_adjust(wspace=0.2, hspace=0.2)
		plt.savefig('tmp.pdf', bbox_inches='tight', pad_inches=0)

	def view_conv_filters(self, conv):
		for i in range(conv.shape[3]):
			plt.figure()
			self.view_conv_filter(conv, idx=i)
		plt.show()

	def view_conv_outputs(self, conv_str, img_idx):
		frames = self.get_single_stack(img_idx)
		# Get convolution output from graph
		conv_out = self.graph.get_tensor_by_name('online/'+conv_str+'/Relu:0')
		# Forward pass up to convolution of choice
		conved_frames = conv_out.eval({self.X_state: frames})
		conved_frames = conved_frames / conved_frames.max()
		batch_size, n_out, _, _ = conved_frames.shape
		n_cols = 4 if conv_str == 'conv2d' else 8
		n_rows = n_out // n_cols

		for i in range(n_out):
			ax = plt.subplot(n_rows, n_cols, i+1)
			ax.imshow(conved_frames[0, i, ...], cmap=plt.cm.gray)
			ax.set_xticks([])
			ax.set_yticks([])
			# ax.set_aspect('equal')
			ax.set_aspect('auto')
		plt.subplots_adjust(wspace=0.2, hspace=0.2)
		plt.savefig('test.pdf', bbox_inches='tight', pad_inches=0)
		plt.show()

	def view_dense_weights(self, fc):
		# plt.imshow(fc.T)
		plt.plot(fc[:, 0])
		plt.show()

	def view_dense_activations(self, dense_str, img_idx):
		frames = self.get_single_stack(img_idx)
		# Get dense output from graph
		if dense_str == 'dense':
			dense_out = self.graph.get_tensor_by_name('online/'+dense_str+'/Relu:0')
		elif dense_str == 'dense_1':
			dense_out = self.graph.get_tensor_by_name('online/'+dense_str+'/BiasAdd:0')
		# Forward pass up to dense activation of choice
		activations = dense_out.eval({self.X_state: frames}).squeeze()
		plt.bar(range(len(activations)), activations)
		plt.show()

	def gaussian_mask(self, x0, y0, sig=5):
		X   = (np.arange(0, 84)-x0)**2 / (2*sig**2)
		Y   = (np.arange(0, 84)-y0)**2 / (2*sig**2)
		return np.exp(-np.add.outer(X, Y))

	def perturbation_saliency_map(self, q_out, img_idx, animate=False):
		# Obtain frame stack and pad
		frames_o = self.get_single_stack(img_idx)
		q_o = q_out.eval({self.X_state: frames_o})

		# Blur
		blurred_frames_o = np.zeros(frames_o.shape)
		for i in range(4):
			blurred_frames_o[0, i, ...] = gaussian_filter(frames_o[0, i, ...], 3)

		P = np.zeros((84, 84))
		# Slide window across frames
		for x in np.arange(0, 84, 5):
			for y in np.arange(0, 84, 5):
				frames = frames_o.copy()
				# Mask
				mask = self.gaussian_mask(x0=x, y0=y)
				for k in range(4):
					frames[0, k] = (1 - mask) * frames[0, k] + mask * blurred_frames_o[0, k]

				# Forward pass
				q = q_out.eval({self.X_state: frames})
				p_local = 0.5 * np.linalg.norm(q_o - q)
				P = P + p_local * mask

		# Up-sample and normalise
		P = zoom(P, zoom=(2.5, 160./84))
		P = (P - P.min()) / (P.max() - P.min())
		col_im = np.array(Image.open('../../Visualisations/'+self.game+'/col_frames/%d.png' % img_idx))
		red = col_im[..., 0]
		col_im[..., 0] = red + ((255-red) * P).round()
		# Return just array for animation
		if animate:
			return col_im
		# Plot
		plt.figure()
		#plt.subplot(131)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(col_im)
		# plt.subplot(132)
		# plt.xticks([])
		# plt.yticks([])
		# plt.imshow(np.array(Image.open('../../Visualisations/'+self.game+'/gray_frames/%d.png' % img_idx)), cmap=plt.cm.gray)
		# plt.subplot(133)
		# plt.xticks([])
		# plt.yticks([])
		# plt.imshow(P, cmap=plt.cm.jet)
		# plt.savefig('../../Visualisations/'+self.game+'/plots/col.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
		# plt.savefig('../../Visualisations/'+self.game+'/plots/strong_policy_2.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
		plt.show()

	def temporal_saliency_map(self, q_out, img_idx, animate=False):
		# Obtain frame stack
		frames_o = self.get_single_stack(img_idx)
		q_o = q_out.eval({self.X_state: frames_o})

		# Blur
		blurred_frames_o = np.zeros(frames_o.shape)
		for i in range(4):
			blurred_frames_o[0, i] = gaussian_filter(frames_o[0, i], 3)

		P = np.zeros((4, 84, 84))
		# Slide window across frames
		for x in np.arange(0, 84, 5):
			for y in np.arange(0, 84, 5):
				# Mask
				mask = self.gaussian_mask(x0=x, y0=y)
				for k in range(4):
					frames = frames_o.copy()
					frames[0, k] = (1 - mask) * frames[0, k] + mask * blurred_frames_o[0, k]
					# Forward pass
					q = q_out.eval({self.X_state: frames})
					p_local = 0.5 * np.linalg.norm(q_o - q)
					P[k] = P[k] + p_local + mask

		# Up-sample and normalise
		tmp = P
		P = np.zeros((4, 210, 160))
		for i in range(4):
			P[i] = zoom(tmp[i], zoom=(2.5, 160./84))
		P = (P - P.min()) / (P.max() - P.min())
		col_ims = []
		for i in range(4):
			idx = img_idx-3+i
			col_im = np.array(Image.open('../../Visualisations/'+self.game+'/col_frames/%d.png' % idx))
			red = col_im[..., 0]
			col_im[..., 0] = red + ((255-red) * P[i]).round()
			col_ims.append(col_im)

		# Return just arrays for animation
		if animate:
			return col_ims
		# Plot
		plt.figure()
		for i in range(4):
			plt.subplot(221+i)
			plt.imshow(col_ims[i])
		plt.show()

	def saliency_game(self, q_out, start_idx, end_idx, temporal=False):
		# Figure
		fig = plt.figure()

		ims = []
		for i in np.arange(start_idx+3, end_idx):
			print(i)
			tic = time.time()
			if temporal:
				P = self.temporal_saliency_map(q_out, i, animate=True)
			else:
				P = self.perturbation_saliency_map(q_out, i, animate=True)
			if isinstance(P, (list,)):
				plots = []
				for i in range(4):
					plots.append(plt.subplot(221+i))
					plots.append(plt.imshow(P[i]))
				ims.append(plots)
			else:
				im = plt.imshow(P, animated=True)
				ims.append([im])
			t = (time.time()-tic)*(end_idx-i)/3600.
			print('%.5f Hours Left' % t)
		ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
		# ani.save(self.game+'_saliency_game.mp4')
		plt.show()

	def saliency_training(self, q_out, img_idx, base_dir, subdir):
		# Fig
		# fig = plt.figure()

		ims = []
		# for i in np.arange(3126, 12500000, 15625):
		for idx, i in enumerate([0, 3126, 4175001, 8346876, 11253126]):
		# for idx, i in enumerate([0, 3126, 3518751, 7050001, 10565626]):
			print(i)
			# Load trained network
			saver = tf.train.import_meta_graph(base_dir+subdir+'/'+subdir+'-%d.meta' % i)
			saver.restore(sess, base_dir+subdir+'/'+subdir+'-%d' % i)

			# Extract graph and relevant tensors
			graph = tf.get_default_graph()
			self.X_state = graph.get_tensor_by_name('state:0')
			q_out = graph.get_tensor_by_name('online/dense_1/MatMul:0')

			# Static priority map for current model
			P = PP.perturbation_saliency_map(q_out, img_idx=img_idx, animate=True)
			# im = plt.imshow(P, animated=True)
			plt.subplot(151+idx)
			plt.xticks([])
			plt.yticks([])
			plt.imshow(P)
			# ims.append([im])
		# ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=3000)
		# ani.save(self.game+'_saliency_training.mp4')
		plt.savefig('../../Visualisations/'+self.game+'/plots/training_improved.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
		plt.show()


if __name__ == "__main__":
	with tf.Session() as sess:
		# Base directory
		base_dir = 'D:/Models/'
		# Load trained network
		run_name = 'breakout'
		# run_name = 'space_invaders'
		# run_name = 'star_gunner_test'
		subdir = run_name
		saver = tf.train.import_meta_graph(base_dir+subdir+'/'+subdir+'-12487501.meta')
		# saver = tf.train.import_meta_graph(base_dir+subdir+'/'+subdir+'-10565626.meta')
		saver.restore(sess, tf.train.latest_checkpoint(base_dir+subdir))
		# Online network variables
		vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		# Extract graph and relevant tensors
		graph = tf.get_default_graph()
		state = graph.get_tensor_by_name('state:0')
		q_out = graph.get_tensor_by_name('online/dense_1/MatMul:0')
		# Post processing class
		PP = PostProcessing(run_name, graph, state)

		# Visualise filters
		PP.view_conv_filter(vars[0].eval(), idx=21)
		# PP.view_conv_filters(vars[0].eval())

		# Visualise convolution activations
		# PP.view_conv_outputs('conv2d', img_idx=1218)

		# Visualise dense weights
		# dense_weights = vars[6].eval()
		# dense_weights_1 = vars[8].eval()
		# PP.view_dense_weights(dense_weights_1)

		# Visualise dense activations
		# PP.view_dense_activations('dense', img_idx=1600)

		# Static priority map for one frame stack
		# PP.perturbation_saliency_map(q_out, img_idx=521)

		# Static priority map for one frame stack
		# PP.temporal_saliency_map(q_out, img_idx=1245)

		# PP.saliency_game(q_out, start_idx=500, end_idx=550, temporal=False)

		# PP.saliency_training(q_out, 516, base_dir, subdir)
