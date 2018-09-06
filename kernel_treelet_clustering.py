import itertools

import numpy as np

from kernel_treelet import kernel_treelet


class kernel_treelet_clustering(kernel_treelet):
	def __init__ (self, kernel=False, number_of_clusters=0, max_sample=500, dropout=0, label_type=None, **kwargs):
		super().__init__(kernel, **kwargs)
		self.max_sample = max_sample
		self.dropout = dropout
		self.tiny_cluster_number = 0
		self.label_type = label_type

		if number_of_clusters is 0:  # Auto-find clust num
			self.number_of_clusters = 2
			self.clustnum_estimate = True
		else:
			self.number_of_clusters = number_of_clusters
			self.clustnum_estimate = False

		# Output Variables Initialization
		self.dataset = None
		self.samp_dataset = None
		self.tree = None
		self.sample_index = None
		self.labels_ = None
		self.svm = None

	def fit (self, X, k=-1):
		self.dataset = np.asmatrix(X)
		if len(self) <= self.max_sample:
			super().fit(self.dataset, k)

			# clustering on sample dataset
			self.tree = self._trl.tree
			if self.clustnum_estimate:
				self.find_clust_num(self._trl.dendrogram_list)

			self.labels_ = self._del_small_clust()
			self.sample_labels = np.array(self.labels_, copy=True)
			self._switch_label_type()
		else:
			self.sample_index = np.arange(self.max_sample)
			self.raw_dataset = np.matrix(self.dataset, copy=True)
			self.dataset = self.dataset[self.sample_index, :]
			self.fit(self.dataset, k)
			from sklearn.svm import SVC
			self.svm = SVC(kernel=self.kernel_name)
			self.svm.fit(self.dataset, self.labels_)
			self.labels_ = self.svm.predict(self.raw_dataset)

	def find_clust_num (self, dendrogram_list):
		# find the first gap with 1
		for i in range(1, self.n - 1):
			if np.abs(dendrogram_list[i - 1] - dendrogram_list[i]) > 1:
				self.number_of_clusters = len(self) - i
				return self.number_of_clusters

	def _del_small_clust (self):
		temp_labels = np.arange(self.n)
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
		if not self.clustnum_estimate:
			if self.dropout is not 0:
				counts = np.unique(temp_labels, return_counts=True)[1]
				tiny_clust_num = (counts < self.dropout).sum()
				if self.tiny_cluster_number < tiny_clust_num:
					self.number_of_clusters += tiny_clust_num - self.tiny_cluster_number
					self.tiny_cluster_number = tiny_clust_num
					self._del_small_clust()
		return temp_labels

	def _switch_label_type (self):
		if self.label_type == None:
			return
		if self.label_type == int:
			temp_dict = {v: k for k, v in enumerate(np.unique(self.labels_), 0)}
			self.labels_ = np.vectorize(temp_dict.__getitem__)(self.labels_)
			return
		else:
			temp_dict = {v: k for k, v in enumerate(np.unique(self.labels_), 0)}
			temp_list = itertools.islice(self.label_type, len(temp_dict))
			tempf = lambda x: temp_list[temp_dict[x]]
			self.labels_ = np.apply_along_axis(np.vectorize(tempf), 0, self.labels_)
			return

	def __len__ (self):
		return self.dataset.shape[0]


KT_clustering = kernel_treelet_clustering
