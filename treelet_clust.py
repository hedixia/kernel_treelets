import numpy as np

from clust import clustering_method
from treelet import treelet


class treelet_clustering(clustering_method):
	def __init__ (self, kernel=False, number_of_clusters=0):
		super().__init__(number_of_clusters)
		self.kernel = kernel if kernel else np.inner

		if number_of_clusters is 0:  # Auto-find clust num
			self.number_of_clusters = 1
			self.clustnum_estimate = True
		else:
			self.number_of_clusters = number_of_clusters
			self.clustnum_estimate = False

		self.tree = None

	def fit (self, X):
		super().fit(X)
		Xlist = self.X.tolist()
		kerf = lambda r_i, r_j: self.kernel(Xlist[r_i], Xlist[r_j])
		self.C = np.vectorize(kerf)(*np.meshgrid(range(self.n), range(self.n), sparse=True))
		if len(self) is 0:
			raise ValueError
		trl = treelet()
		trl.fit(self.C)
		self.tree = trl.tree
		if self.clustnum_estimate:
			self.find_clust_num(trl.dendrogram_list)
		temp_labels = list(range(self.n))
		for i in range(self.n - self.number_of_clusters):
			temp_labels[self.tree[i][1]] = self.tree[i][0]
		for i in range(self.n):
			current = i
			while current != temp_labels[current]:
				current = temp_labels[current]
			ending = current
			current = i
			while current != ending:
				temp_labels[current] = ending
				current = temp_labels[current]
		self.labels = temp_labels
		self._l2c()

	def find_clust_num (self, dendrogram_list):
		# find the first gap with 1
		for i in range(len(self) - 1):
			if i is 0:
				continue
			else:
				if np.abs(dendrogram_list[i - 1] - dendrogram_list[i]) > 1:
					self.number_of_clusters = len(self) - i
					break
