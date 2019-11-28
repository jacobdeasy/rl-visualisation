#!/usr/bin/env python3
import matplotlib.pyplot as plt, numpy as np, os, pandas as pd

from sklearn.preprocessing import scale
from sklearn.cluster import KMeans, AgglomerativeClustering


# Directory
game = 'space_invaders'
dir2use = os.path.join(os.pardir, 'Visualisations', game)

# Read in data
time = pd.read_csv(os.path.join(dir2use, 'time_num.tsv'), sep='\t', header=None).values
tsne_result = pd.read_csv(os.path.join(dir2use, 'tSNE_perp50.tsv'), sep='\t', header=None).values
X = np.concatenate((tsne_result, time), axis=1)

# Normalise data
X = scale(X)

# Plot
plt.figure()
plt.scatter(x=X[:, 0], y=X[:, 1], s=0.9, c=X[:, 2], cmap=plt.cm.jet)
plt.colorbar()
plt.axis('off')
plt.title(game.capitalize()+' t-SNE')

# K-means
kmeans = KMeans(n_clusters=15)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Spectral clustering
model = AgglomerativeClustering(n_clusters=15)
labels = model.fit_predict(X)

# Plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=0.9, c=y_kmeans)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', alpha=0.5)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=0.9, c=labels)
plt.show()
