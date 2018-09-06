import numpy as np

import treelet


class kernel_treelet:
	def __init__ (self, kernel=False, **kwargs):
		# Input Variables
		self.kernel_name = kernel
		self._ker_ = self._kernel(kernel)
		self.__dict__.update(kwargs)

		# Intermediate Variables
		self._trl = treelet.treelet()

		# Output Variables Initialization
		self.__X = None
		self.A_0 = None
		self.A_k = None
		self.L_k = None
		self.Delta_k = None

	def __len__ (self):
		return self._trl.n

	@property
	def n (self):
		return self._trl.n

	@property
	def transform_list (self):
		return self._trl.transform_list

	def fit (self, X, k=-1):
		self.__X = np.asmatrix(X)
		n = self.__X.shape[0]
		if k < 0:
			k += n
		A_0 = np.fromfunction(self._kernel_matrix_function, shape=(n, n), dtype=int)
		self.A_0 = np.matrix(A_0)
		self._trl.fit(A_0)
		A_k = self.transform(self.transform(self.A_0.getT(), k).getT(), k)
		self.A_k = A_k
		self.L_k = self._decomp(A_k)
		self.Delta_k = self.transform(np.identity(A_0.shape[0])) * self.L_k

	def transform (self, v, k=1):
		v = np.matrix(v)
		for i in range(self.n - k):
			(scv, cgs, cos_val, sin_val) = self.transform_list[i]
			temp_scv = cos_val * v[:, scv] - sin_val * v[:, cgs]
			temp_cgs = sin_val * v[:, scv] + cos_val * v[:, cgs]
			v[:, scv] = temp_scv
			v[:, cgs] = temp_cgs
		return v

	def _decomp (self, M):
		M = np.asmatrix(M)
		farray = [I[1] for I in self.transform_list] + [self._trl.root]

		# rearrange the matrix with the order of tree
		tempf = lambda x, y: M[farray[x], farray[y]]
		M = np.asmatrix(np.fromfunction(np.vectorize(tempf), shape=M.shape, dtype=int))

		# cholesky decomposition
		L = np.sqrt(np.diag(np.diag(M)))

		# rearrange the matrix back to origional order
		barray = np.zeros(len(farray), dtype=int)
		for i in range(len(farray)):
			barray[farray[i]] = i
		L = L[barray, :]
		return L

	def _kernel_matrix_function (self, x, y):
		return self._ker_(self.__X[x, :], self.__X[y, :])

	def _kernel (self, kernel):  # return a kernel function f:SxS->R
		if kernel == "rbf":
			kernel = self._rbf
		if kernel == "poly":
			kernel = self._poly
		return kernel

	@property
	def __twosigmasq (self):
		if not hasattr(self, "_twosigmasq"):
			self._twosigmasq = 2 * self.sigma * self.sigma
		return self._twosigmasq

	def _rbf (self, x, y):  # Radial Basis Function Kernel
		diff = np.linalg.norm(np.asarray(x) - np.asarray(y), axis=-1)
		return np.exp(- diff * diff / self.__twosigmasq)

	def _poly (self, x, y): # Polynomial Kernel
		return ((np.asarray(x) * np.asarray(y)).sum(axis=-1) + self.d) ** self.pow


KT = kernel_treelet
