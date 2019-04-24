import pickle
import time
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
from kernel_treelets_clustering import kernel_treelets_clustering

timelist = []
timelist.append(time.time())
graph = np.genfromtxt(r"datasets\\facebook_combined.txt", dtype=np.uint16)
adjmat_int8 = coo_matrix((np.ones(graph.shape[0], dtype=np.uint16), graph.T), shape=(4039, 4039), dtype=np.int8).toarray()
adjmat_int8 = adjmat_int8 + adjmat_int8.T
adjmat = np.array(adjmat_int8, dtype=float)
adjmat[np.diag_indices_from(adjmat)] = 1045


class network_kernel:
	def __init__ (self, diag):
		self.diag = diag

	def __call__ (self, X):
		X = X[:, np.asarray(X).max(axis=0) > 1]
		print(X)
		X = X.copy()
		X[np.diag_indices_from(X)] = self.diag
		return X

def conf_mat (label):
	augmented_labels = np.asarray(label).reshape(-1, 1)
	comp_mat = adjmat_int8 * 2 + (augmented_labels == augmented_labels.T)
	cm = np.bincount(comp_mat.flatten(), minlength=4)
	return cm

core = 1045
timelist.append(time.time())
ktc = kernel_treelets_clustering(network_kernel(core), number_of_clusters=30, max_sample=4039, verbose=True)
ktc.fit(adjmat)
timelist.append(time.time())
print(ktc.labels_)
print(Counter(ktc.labels_))
print(np.diff(timelist))
with open(r"C:\Users\Hedi Xia\Desktop\ktc_fb" + str(core), "wb") as output:
	pickle.dump(ktc, output)





print(conf_mat(ktc.labels_))

L = []
labels = np.arange(4039, dtype=int)
for i in range(4039):
	confmat = conf_mat(labels)
	print("conf iteration:", i, confmat)
	L.append(confmat)
	try:
		labels[ktc._trl.tree[i][1]] = ktc._trl.tree[i][0]
		for j in range(4039):
			current = j
			while current != labels[current]:
				current = labels[current]
			ending = current
			current = j
			while current != ending:
				labels[current] = ending
				current = labels[current]
	except Exception:
		pass
L = np.transpose(L)
L[1] = L[1] - 4039
tp = L[3] / (L[3] + L[2])
fp = L[1] / (L[0] + L[1])

def find_val (model_type, **kwargs):
	L = []
	for i in range(1, 4039):
		model = model_type(i, **kwargs)
		model.fit(np.array(adjmat_int8, dtype=float))
		mconfmat = conf_mat(model.labels_)
		if mconfmat[1] == 0:
			break
		L.append(mconfmat)
		print("conf iteration:", i, mconfmat, model.__class__.__name__)
	L = np.transpose(L)
	L[1] = L[1] - 4039
	tp = L[3] / (L[3] + L[2])
	fp = L[1] / (L[0] + L[1])
	return fp, tp


plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("FB Network Data ROC curve (core=1045)")
methods = [plt.plot(fp, tp)]
names = ["KT"]
auc = [np.trapz(tp, fp)]
plt.legend([i[0] for i in methods], [names[i] + " AUC = {:.3f}".format(auc[i]) for i in range(len(methods))])

plt.show()
