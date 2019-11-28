#!/usr/bin/env python3
from matplotlib import pyplot as plt


class Slicer(object):
	"""docstring for Slicer"""
	def __init__(self):
		super(Slicer, self).__init__()

	def multi_slice_viewer(self, volume, gray=True):
		self.remove_keymap_conflicts({'j', 'k'})
		fig, ax = plt.subplots()
		ax.volume = volume
		ax.index = 0
		if gray:
			ax.imshow(volume[..., ax.index], cmap=plt.cm.gray)
		else:
			ax.imshow(volume[..., ax.index])
		fig.canvas.mpl_connect('key_press_event', self.process_key)

	def process_key(self, event):
		fig = event.canvas.figure
		ax = fig.axes[0]
		if event.key == 'j':
			self.previous_slice(ax)
		elif event.key == 'k':
			self.next_slice(ax)
		fig.canvas.draw()

	def previous_slice(self, ax):
		"""Go to the previous slice."""
		volume = ax.volume
		ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
		ax.images[0].set_array(volume[..., ax.index])

	def next_slice(self, ax):
		"""Go to the next slice."""
		volume = ax.volume
		ax.index = (ax.index + 1) % volume.shape[2]
		ax.images[0].set_array(volume[..., ax.index])

	def remove_keymap_conflicts(self, new_keys_set):
		for prop in plt.rcParams:
			if prop.startswith('keymap.'):
				keys = plt.rcParams[prop]
				remove_list = set(keys) & new_keys_set
				for key in remove_list:
					keys.remove(key)
