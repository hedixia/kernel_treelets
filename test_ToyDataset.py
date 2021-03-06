"""
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

This example shows characteristics of different
clustering algorithms on datasets that are "interesting"
but still in 2D. With the exception of the last dataset,
the parameters of each of these dataset-algorithm pairs
has been tuned to produce good clustering results. Some
algorithms are more sensitive to parameter values than
others.

The last dataset is an example of a 'null' situation for
clustering: the data is homogeneous, and there is no good
clustering. For this example, the null dataset uses the
same parameters as the dataset in the row above it, which
represents a mismatch in the parameter values and the
data structure.

While these examples give some intuition about the
algorithms, this intuition might not apply to very high
dimensional data.
"""
print(__doc__)

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from kernel_treelets_clustering import kernel_treelets_clustering
from sklearn.decomposition import PCA

cmaps = [
	'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
	'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
	'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']


def high_dim_plot (M, c):
	pca = PCA(n_components=2)
	points = np.transpose(pca.fit_transform(M))
	# points = np.asarray(M[:,:2])
	sp = plt.subplot(111)
	sp.scatter(points[0], points[1], c=c)
	plt.show()


np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3}

datasets = [
	(noisy_circles, {'damping': .77, 'preference': -240, 'quantile': .2, 'n_clusters': 2, 'n': 2}),
	(noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2, 'n': 2}),
	(varied, {'eps': .18, 'n_neighbors': 2, 'n': 9}),
	(aniso, {'eps': .15, 'n_neighbors': 2, 'n': 6}),
	(blobs, {'n': 3}),
	(no_structure, {'n': 3})
]

Chart = []
namelist = []

for i_dataset, (dataset, algo_params) in enumerate(datasets):
	# update parameters with dataset-specific values
	params = default_base.copy()
	params.update(algo_params)

	X, y = dataset
	print(X.shape)
	# normalize dataset for easier parameter selection
	X = StandardScaler().fit_transform(X)

	# estimate bandwidth for mean shift
	bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

	# connectivity matrix for structured Ward
	connectivity = kneighbors_graph(
		X, n_neighbors=params['n_neighbors'], include_self=False)
	# make connectivity symmetric
	connectivity = 0.5 * (connectivity + connectivity.T)

	# ============
	# Create cluster objects
	# ============
	ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
	two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
	ward = cluster.AgglomerativeClustering(
		n_clusters=params['n_clusters'], linkage='ward',
		connectivity=connectivity)
	spectral = cluster.SpectralClustering(
		n_clusters=params['n_clusters'], eigen_solver='arpack',
		affinity="nearest_neighbors")
	dbscan = cluster.DBSCAN(eps=params['eps'])
	affinity_propagation = cluster.AffinityPropagation(
		damping=params['damping'], preference=params['preference'])
	average_linkage = cluster.AgglomerativeClustering(
		linkage="average", affinity="cityblock",
		n_clusters=params['n_clusters'], connectivity=connectivity)
	birch = cluster.Birch(n_clusters=params['n_clusters'])
	gmm = mixture.GaussianMixture(
		n_components=params['n_clusters'], covariance_type='full')
	kt = kernel_treelets_clustering(kernel='linear', number_of_clusters=params['n_clusters'], label_type=int, )

	kte1000 = kernel_treelets_clustering(kernel='rbf', sigma=0.1,
	                                     max_sample=1000, label_type=int,
	                                     number_of_clusters=params['n'],
	                                     )

	kte1500 = kernel_treelets_clustering(kernel='rbf', sigma=0.1,
	                                     max_sample=1500, label_type=int,
	                                     number_of_clusters=params['n'],
	                                     )

	ktp = kernel_treelets_clustering(kernel='poly', sigma=1, degree=2, coef0=0.3, number_of_clusters=params['n_clusters'], label_type=int, )

	other_alg = [
		('MiniBatchKMeans', two_means),
		#('AffinityPropagation', affinity_propagation),
		('MeanShift', ms),
		('SpectralClustering', spectral),
		('Ward', ward),
		('AgglomerativeClustering', average_linkage),
		('DBSCAN', dbscan),
		#('Birch', birch),
		('GaussianMixture', gmm),

	]

	KT_alg = [
		('KTrbf', kte1500),
		('KT', kt),
		('KTpoly', ktp),
	]

	KT_num = [
		('KT' + str(n), kernel_treelets_clustering(kernel='rbf', sigma=0.1,
		                                           max_sample=n, label_type=int,
		                                           number_of_clusters=params['n'],
		                                           )) for n in [50, 100, 200, 300, 500, 800, 1000, 1200, 1499, 1500]
	]

	clustering_algorithms = KT_alg + other_alg
	# clustering_algorithms = KT_num

	for iter_ in range(len(clustering_algorithms)):
		name, algorithm = clustering_algorithms[iter_]
		t0 = time.time()

		# catch warnings related to kneighbors_graph
		with warnings.catch_warnings():
			warnings.filterwarnings(
				"ignore",
				message="the number of connected components of the " +
				        "connectivity matrix is [0-9]{1,2}" +
				        " > 1. Completing it to avoid stopping the tree early.",
				category=UserWarning)
			warnings.filterwarnings(
				"ignore",
				message="Graph is not fully connected, spectral embedding" +
				        " may not work as expected.",
				category=UserWarning)
			algorithm.fit(X)
		t1 = time.time()
		if hasattr(algorithm, 'labels_'):
			y_pred = algorithm.labels_.astype(np.int)
		else:
			y_pred = algorithm.predict(X)

		plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
		if i_dataset == 0:
			plt.title(iter_, size=25)
			namelist.append(str(iter_) + " - " + name)

		colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
		                                     '#f781bf', '#a65628', '#984ea3',
		                                     '#999999', '#e41a1c', '#dede00']),
		                              int(max(y_pred) + 1))))
		plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

		plt.xlim(-2.5, 2.5)
		plt.ylim(-2.5, 2.5)
		plt.xticks(())
		plt.yticks(())

		Chart.append('%.3f' % (t1 - t0))
		plot_num += 1
		if isinstance(algorithm, kernel_treelets_clustering):
			# high_dim_plot(algorithm.Delta_k, algorithm.sample_labels)
			print(algorithm.number_of_clusters)
			from collections import Counter

			print(Counter(algorithm._labels_))

plt.show()

Chart = np.asarray(namelist + Chart).reshape(7, -1)

np.savetxt("temp.csv", np.transpose(Chart), fmt='%s', delimiter=' & ')
print(Chart)