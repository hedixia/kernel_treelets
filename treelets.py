import numpy as np


def jacobi_rotation (M, k, l, tol=10 ** (-11)):
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


class treelets:
	def __init__ (self, verbose=False):
		self.n = 0
		self.verbose = verbose
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
		self.M_ = np.asarray(X)
		self.n = self.M_.shape[0]
		self.active = np.ones(self.n, dtype=bool)
		self.max_row = np.zeros(self.n, dtype=int)
		self._rotate(self.n - 1)
		self.root = self.max_row[np.nonzero(self.active)[0][0]]

	def _rotate (self, multi=False):
		if multi:
			for i in range(multi):
				self._rotate()
				if self.verbose:
					print("rotation: ", i, "\tcurrent: ", self.current)
			self.dfrk = [self.transform_list[i][1] for i in range(self.n - 1)]
			self.dfrk.append(self.transform_list[-1][0])
		else:
			(p, q) = self._find()
			(cos_val, sin_val) = jacobi_rotation(self.M_, p, q)
			self._record(p, q, cos_val, sin_val)

	def _find (self):
		self.active_list = np.nonzero(self.active)[0]
		if self.transform_list:
			k, l, *_ = self.current
			for i in self.active_list:
				self._maintainance(i, k, l)
		else:
			self.max_row_val = np.zeros(self.n)
			for i in self.active_list:
				self._max(i)

		k = np.argmax(self.max_row_val * self.active)
		v = self.max_row_val[k]
		self.dendrogram_list.append(np.log(v))
		return self.max_row[k], k

	def _maintainance (self, i, k, l):
		if i in (k, l):
			self._max(i)
		if self.M_[self.max_row[i], i] < self.M_[l, i]:
			self.max_row[i] = l
		if self.M_[self.max_row[i], i] < self.M_[k, i]:
			self.max_row[i] = k
		if self.max_row[i] in (k, l):
			self._max(i)

	def _max (self, col_num):
		temp = np.abs(self.M_[col_num]) * (self.active - 0.5)
		temp[col_num] = -1
		self.max_row[col_num] = np.argmax(temp)
		self.max_row_val[col_num] = self.M_[self.max_row[col_num], col_num]

	def _record (self, l, k, cos_val, sin_val):
		if self.M_[l, l] < self.M_[k, k]:
			self.current = (k, l, cos_val, sin_val)
		else:
			self.current = (l, k, cos_val, sin_val)

		self.transform_list.append(self.current)
		self.active[self.current[1]] = False
