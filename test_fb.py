import pickle
import time
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix

from kernel_treelet_clustering import kernel_treelet_clustering

timelist = []
timelist.append(time.time())
graph = np.genfromtxt("facebook_combined.txt", dtype=np.uint16)
adjmat = coo_matrix((np.ones(graph.shape[0], dtype=np.uint16), graph.T), shape=(4039, 4039), dtype=float).todense()
adjmat = adjmat + adjmat.T
adjmat[np.diag_indices_from(adjmat)] = 5


class network_kernel:
	def __init__ (self, diag):
		self.diag = diag

	def __call__ (self, X):
		X = X[:, np.asarray(X).max(axis=0) > 1]
		print(X)
		X = X.copy()
		X[np.diag_indices_from(X)] = self.diag
		return X

core = 5
timelist.append(time.time())
ktc = kernel_treelet_clustering(network_kernel(core), number_of_clusters=30, max_sample=4039, verbose=True)
ktc.fit(adjmat)
timelist.append(time.time())
print(ktc.labels_)
print(Counter(ktc.labels_))
print(np.diff(timelist))
with open(r"C:\Users\Hedi Xia\Desktop\temp\ktc_fb"+str(core), "wb") as output:
	pickle.dump(ktc, output)
