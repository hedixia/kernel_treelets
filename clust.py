import numpy as np


class clustering_method:
	def __init__ (self, number_of_clusters=0):
		self.number_of_clusters = number_of_clusters
		self.labels = []
		self.clusters = {}
		self.sorted_labels = False
		self.DataMatrix = False
		self.n = 0

	def __len__ (self):
		return self.n

	@property
	def labels_ (self):
		if not self.sorted_labels:
			self.sorted_labels = [self.labels[i] for i in sorted(self.labels)]
		return self.sorted_labels

	def fit (self, X):
		self.X = np.matrix(X)
		self.n = self.X.shape[0]

	def _l2c (self):
		for i in range(self.n):
			self.clusters.setdefault(self.labels[i], []).append(i)

	def _c2l (self):
		self.labels = [0] * self.n
		for i in self.clusters:
			for j in self.clusters[i]:
				self.labels[j] = i
