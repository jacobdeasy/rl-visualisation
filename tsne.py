#!/usr/bin/env python3
import numpy as np, os, pandas as pd

from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Directory
game = 'space_invaders'
dir2use = os.path.join(os.pardir, 'Visualisations', game)

# Read in data
dense_mat = pd.read_csv(os.path.join(dir2use, 'hidden_activations.tsv'), sep='\t', header=None)
max_q = pd.read_csv(os.path.join(dir2use, 'q_values.tsv'), sep='\t', header=None).max(axis=1)
time = pd.read_csv(os.path.join(dir2use, 'time_num.tsv'), sep='\t', header=None)

# t-SNE
calc_tSNE = False
perp = 50.0
N = dense_mat.shape[0]

if calc_tSNE:
	# PCA on dense_mat
	pca = PCA(n_components=50)
	pca_result = pca.fit_transform(dense_mat)
	print('Proportion variance explained:', sum(pca.explained_variance_ratio_))
	# tsne on pca_result
	tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000, verbose=2)
	tsne_result = tsne.fit_transform(pca_result[:N, :])
	np.savetxt(os.path.join(dir2use, f'/tSNE_perp{perp}.tsv'), tsne_result, delimiter='\t')
else:
	tsne_result = pd.read_csv(os.path.join(dir2use, f'tSNE_perp{perp}.tsv'), sep='\t', header=None).values

# Colour by time point
plt.figure()
plt.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], s=0.01, c=max_q, cmap=plt.cm.jet)
plt.axis('off')

# Read in images
frames = []
for i in range(N):
	frames.append(np.array(Image.open(os.path.join(dir2use, 'col_frames', f'{i}.png'))))

# HOVER PLOT
fig = plt.figure()
ax = fig.add_subplot(111)
line = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], s=0.9, c=max_q[:N], cmap=plt.cm.jet)
fig.colorbar(line, ax=ax)

# Create the annotations box
im = OffsetImage(frames[0], zoom=0.5)
xybox = (50., 50.)
ab = AnnotationBbox(im, (0, 0), xybox=xybox, xycoords='data', boxcoords='offset points', pad=0.0)

# Add it to axes and make it visible
ax.add_artist(ab)
ab.set_visible(False)

def hover(event):
	# If the mouse is over the scatter points
	if line.contains(event)[0]:
		# Find out the index within the array from the event
		ind = line.contains(event)[1]['ind'][0]
		print(ind)
		# Get the figure size
		w, h = fig.get_size_inches() * fig.dpi
		# If event occurs in the top or right quadrant of the figure,
		# change the annotation box position relative to mouse.
		ws = (event.x > w/2.) * -1 + (event.x <= w/2.) 
		hs = (event.y > h/2.) * -1 + (event.y <= h/2.)
		ab.xybox = (xybox[0]*ws, xybox[1]*hs)
		# Make annotation box visible
		ab.set_visible(True)
		# Place it at the position of the hovered scatter point
		ab.xy = (tsne_result[ind, 0], tsne_result[ind, 1])
		# Set the image corresponding to that point
		im.set_data(frames[ind])
	else:
		#if the mouse is not over a scatter point
		ab.set_visible(False)
	fig.canvas.draw_idle()

# Add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)
plt.title(game.capitalize()+' t-SNE')
plt.axis('off')
plt.show()
