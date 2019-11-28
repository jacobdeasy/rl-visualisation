#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

from sys import platform
from matplotlib import pyplot as plt


def get_layer_names():
	# Name of final operation at each layer
	names = [
		'online/conv2d/Conv2D',
		'online/conv2d_1/Conv2D',
		'online/conv2d_2/Conv2D',
		'online/dense/MatMul',
		'online/dense_1/MatMul']

	return names


def subplot4(image):
	for i in range(4):
		ax = plt.subplot(221+i)
		ax.imshow(image[0, i, ...], cmap=plt.cm.gray)
		ax.set_xticks([])
		ax.set_yticks([])
		tmp = 3-i
		ax.set_title('t-%d' % tmp)
		ax.set_aspect('auto')
	plt.subplots_adjust(wspace=0.2, hspace=0.2)


def optimize_image(layer_id=None, feature=0, num_iterations=100, show_progress=False):
	# Set graph
	graph = tf.get_default_graph()

	# Tensor for input image
	img_tensor_name = 'state:0'
	img_tensor = graph.get_tensor_by_name(img_tensor_name)

	# Get list of final operations at each layer
	layer_names = get_layer_names()

	# Create the loss function that must be maximised
	if layer_id is None:
		# If no layer_id is provided, maximise the final layer feature
		loss = graph.get_tensor_by_name(layer_names[-1] + ':0')[0, feature]

	else:
		# Get name of layer operator
		layer_name = layer_names[layer_id]

		# Reference to tensor that is output by the operator
		tensor = graph.get_tensor_by_name(layer_name + ':0')

		# Loss function is average of all tensor values for given feature.
		loss = tf.reduce_mean(tensor[0, feature, ...])

	# Get gradient of loss function with respect to the input image via the chain rule.
	gradient = tf.gradients(loss, img_tensor)

	# Random initial frames
	image = np.random.uniform(size=(1, 4, 84, 84))+127.5

	# Iterations to optimize the image
	for i in range(num_iterations):
		# Calculate gradient and the loss value
		grad, loss_value = sess.run([gradient, loss], feed_dict={img_tensor_name: image})

		# Convert to gradient array
		grad = np.array(grad)[0]

		# Calculate the step-size for updating the image.
		# step_size = 1.0 / (grad.std() + 1e-8)
		step_size = 100

		# update the image by adding the scaled gradient (gradient ASCENT)
		image += step_size * grad

		# Clip pixel-values to be between 0 and 255
		image = np.clip(image, 0.0, 255.0)

		if show_progress:
			print('Iteration:', i)
			# Print statistics for the gradient.
			msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
			print(msg.format(grad.min(), grad.max(), step_size))
			# Print the loss-value.
			print("Loss:", loss_value)
			# Newline.
			print()

			# Plot
			subplot4(image)
			plt.pause(0.01)

	return image


def loop_act_max(layer_id, layer_length, num_iter):
	# Loop through layer
	for i in range(layer_length):
		print('Kernel index: ', i)
		opt_img = optimize_image(layer_id=layer_id, feature=i, num_iterations=num_iter)
		plt.figure()
		subplot4(opt_img)
		plt.savefig('../../Visualisations/'+run_name+'/activation_maximisation/layer%d/%d' % (layer_id, i))


if __name__ == "__main__":
	with tf.Session() as sess:
		# Base directory
		base_dir = 'D:/Models/'

		# Load trained network
		run_name = 'breakout'
		subdir = run_name
		saver = tf.train.import_meta_graph(base_dir+subdir+'/'+subdir+'-12206251.meta')
		saver.restore(sess, tf.train.latest_checkpoint(base_dir+subdir))

		# loop_act_max(layer_id=1, layer_length=64, num_iter=1000)
		# for i, iter_num in enumerate([0, 1000, 2000, 5000]):
		# 	opt_img = optimize_image(0, 14, iter_num, show_progress=False)
		# 	ax = plt.subplot(141+i)
		# 	ax.imshow(opt_img[0, 3, ...], cmap=plt.cm.gray)
		# 	ax.set_xticks([])
		# 	ax.set_yticks([])
		# 	ax.set_title('Iteration=%d' % iter_num)
		# 	ax.set_aspect('equal')
		# 	plt.subplots_adjust(wspace=0.2, hspace=0.2)

		opt_img = optimize_image(3, 14, 5000, show_progress=False)
		subplot4(opt_img)

		plt.savefig('AM_noise.pdf', bbox_inches='tight', pad_inches=0)
		plt.show()
