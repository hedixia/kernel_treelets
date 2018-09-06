import numpy as np


def jacobi_rotation (M, k, l, tol=0.00000000001):
	"""
	input: numpy matrix for rotation M, two different row number k and l 
	output: cos and sin value of rotation 
	change: M is inplace changed
	"""

	# rotation matrix calc
	if M[k, l] + M[l, k] < tol:
		cos_val = 1
		sin_val = 0
	else:
		b = (M[l, l] - M[k, k]) / (M[k, l] + M[l, k])
		tan_val = (1 if b >= 0 else -1) / (abs(b) + np.sqrt(b * b + 1))  # |tan_val| < 1
		cos_val = 1 / (np.sqrt(tan_val * tan_val + 1))  # cos_val > 0
		sin_val = cos_val * tan_val  # |cos_val| > |sin_val|

	# right multiplication by jacobian matrix
	temp1 = M[k, :] * cos_val - M[l, :] * sin_val
	temp2 = M[k, :] * sin_val + M[l, :] * cos_val
	M[k, :] = temp1
	M[l, :] = temp2

	# left multiplication by jacobian matrix transpose
	temp1 = M[:, k] * cos_val - M[:, l] * sin_val
	temp2 = M[:, k] * sin_val + M[:, l] * cos_val
	M[:, k] = temp1
	M[:, l] = temp2

	return cos_val, sin_val


class treelet:
	def __init__ (self, psi=False):
		self.psi = psi if psi else treelet.default_psi
		self.n = 0
		self.__X = None
		self.M_ = None
		self.max_row = None
		self.root = None
		self.dfrk = None
		self._tree = None
		self._layer = None
		self.active = None
		self.active_list = None
		self.transform_list = []
		self.dendrogram_list = []

	def __len__ (self):
		return self.n

	# Treelet Tree
	@property
	def tree (self):
		if self._tree is None:
			self._tree = [I[0:2] for I in self.transform_list]
		return self._tree

	@property
	def layer (self):
		if self._layer is None:
			self._layer = np.ones(self.n, dtype=int)
			for merging in self.tree:
				self._layer[merging[0]] += self._layer[merging[1]]
		return self._layer

	def fit (self, X):
		self.__X = np.asmatrix(X)
		self.M_ = np.asmatrix(np.fromfunction(np.vectorize(self.phi), self.__X.shape, dtype=int))
		self.n = self.__X.shape[0]
		self.active = np.ones(self.n, dtype=bool)
		self.max_row = np.zeros(self.n, dtype=int)
		self._rotate(self.n - 1)
		self.root = self.max_row[np.nonzero(self.active)[0][0]]

	def phi (self, x, y):
		return self.psi(self.__X[x, y], self.__X[x, x], self.__X[y, y])

	def _rotate (self, multi=False):
		if multi:
			for _ in range(multi):
				self._rotate()
			self.dfrk = [self.transform_list[i][1] for i in range(self.n - 1)]
			self.dfrk.append(self.transform_list[-1][0])
		else:
			(p, q) = self._find()
			(cos_val, sin_val) = jacobi_rotation(self.__X, p, q)
			self._record(p, q, cos_val, sin_val)

	def _find (self):
		self.active_list = np.nonzero(self.active)[0]
		if self.transform_list:
			k, l, *_ = self.current
			for i in self.active_list:
				if i in (k, l):
					self._max(i)
				if self.M_[self.max_row[i], i] < self.M_[l, i]:
					self.max_row[i] = l
				if self.M_[self.max_row[i], i] < self.M_[k, i]:
					self.max_row[i] = k
				if self.max_row[i] in (k, l):
					self._max(i)
		else:
			self.max_row_val = np.zeros(self.n)
			for i in self.active_list:
				self._max(i)

		k = np.argmax(self.max_row_val * self.active)
		v = self.max_row_val[k]
		self.dendrogram_list.append(np.log(v))
		return self.max_row[k], k

	def _max (self, col_num):
		temp_max_row = 0
		max_temp = 0
		for i in self.active_list:
			if i == col_num:
				continue
			temp = self.M_[i, col_num]
			if temp >= max_temp:
				temp_max_row = i
				max_temp = temp
		self.max_row[col_num] = temp_max_row
		self.max_row_val[col_num] = max_temp

	def _record (self, l, k, cos_val, sin_val):
		if self.__X[l, l] < self.__X[k, k]:
			self.current = (k, l, cos_val, sin_val)
		else:
			self.current = (l, k, cos_val, sin_val)

		sca_ind = self.current[0]
		dif_ind = self.current[1]

		temp = np.fromfunction(np.vectorize(lambda x, y: self.phi(sca_ind, y)), (1, self.n), dtype=int)
		self.M_[sca_ind, :] = temp
		self.M_[:, sca_ind] = np.transpose(temp)

		self.transform_list.append(self.current)
		self.active[dif_ind] = False

	@staticmethod
	def default_psi (x, y, z):
		return np.abs(x) / np.sqrt(np.abs(y * z))
