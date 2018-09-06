import numpy as np
from scipy.sparse import coo_matrix

from treelet import treelet


class treelet_dimred(treelet):
	def __init__ (self, t=0):
		super().__init__()
		self.t = t
		self.psi = lambda x, y, z: abs(x) / np.sqrt(np.abs(y * z)) + abs(x) * self.t

		self.mean_ = None
		self.cov = None

	def fit (self, X):
		self.dataset_ = np.matrix(X)
		self.mean_ = np.mean(X, axis=0)
		self.cov = np.cov(X.getT())
		super().fit(self.cov)

	# Treelet Transform
	def transform (self, v, k=False, epsilon=0):
		v = np.matrix(v) - self.mean_
		k = k if k else 1
		for iter in range(self.n - k):
			(scv, cgs, cos_val, sin_val) = self.transform_list[iter]
			temp_scv = cos_val * v[:, scv] - sin_val * v[:, cgs]
			temp_cgs = sin_val * v[:, scv] + cos_val * v[:, cgs]
			v[:, scv] = temp_scv
			v[:, cgs] = temp_cgs
		if epsilon == 0:
			return v, None
		else:
			scaling_part = np.concatenate([v[:, self.dfrk[i]] for i in range(self.n - k, self.n)], axis=1)
			difference_part = np.concatenate([v[:, self.dfrk[i]] for i in range(self.n - k)], axis=1)
			difference_mat = coo_matrix(abs(difference_part) > epsilon).multiply(difference_part)
			return scaling_part, difference_mat

	def inverse_transform (self, scaling_part, difference_matrix=False):
		scaling_part = np.matrix(scaling_part)
		k = self.n - scaling_part.shape[1]
		v = np.matrix(np.zeros((scaling_part.shape[0], self.n)))
		for i in range(k, self.n):
			v[:, self.dfrk[i]] = scaling_part[:, i - k]
		for iter in reversed(self.transform_list):
			(scv, cgs, cos_val, sin_val) = iter
			temp_scv = cos_val * v[:, scv] + sin_val * v[:, cgs]
			temp_cgs = -sin_val * v[:, scv] + cos_val * v[:, cgs]
			v[:, scv] = temp_scv
			v[:, cgs] = temp_cgs
		if difference_matrix:
			for i in range(k):
				v[:, self.dfrk[i]] += difference_matrix[:, i]
		return v + self.mean_

	#def

	def cluster (self, k):
		clust_list = list(range(self.n))
		for i in range(self.n - k, -1, -1):
			clust_list[self.transform_list[i][1]] = clust_list[self.transform_list[i][0]]
		return clust_list

	def components_ (self, k):
		return self.transform(np.identity(self.n) + self.mean_, k=k)[0]

	__call__ = transform
