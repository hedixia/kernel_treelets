from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
import time
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix

from kernel_treelets_clustering import kernel_treelets_clustering

timelist = []
timelist.append(time.time())
dataset = np.genfromtxt(r"datasets/Data_Cortex_Nuclear.csv", delimiter=',')[1:]
dataset = dataset[:, ~np.all(np.isnan(dataset), axis=0)]
dataset = (dataset - np.nanmean(dataset, axis=0)) / np.nanstd(dataset, axis=0)
imp_dataset = Imputer().fit_transform(dataset)
labels = np.genfromtxt(r"datasets/Data_Cortex_Nuclear.csv", delimiter=',', dtype=str)[1:, -1]


class nan_kernel:
	def __init__ (self, sigma=1):
		self.gamma = 1 / (2 * sigma * sigma)

	def __call__ (self, X):
		a, b = X.shape
		X = np.asarray(X).reshape((a, 1, b))
		return np.nanmean(X * X.swapaxes(1, 2), axis=0)


class nan_kernel_rbf:
	def __init__ (self, sigma=1):
		self.gamma = 1 / (2 * sigma * sigma)

	def __call__ (self, X):
		a, b = X.shape
		X = np.asarray(X).reshape((a, 1, b))
		diff = np.nanmean((X - X.swapaxes(1, 2)) ** 2, axis=0)
		return np.exp(- diff * diff * self.gamma)


ktc = kernel_treelets_clustering(nan_kernel(1), number_of_clusters=5, max_sample=2000, verbose=True)
ktc.fit(dataset)
print(ktc.labels_)
print(Counter(ktc.labels_))

timelist.append(time.time())
ktc = kernel_treelets_clustering(nan_kernel_rbf(0.125), number_of_clusters=15, max_sample=2000, verbose=True)
ktc.fit(dataset.T)
timelist.append(time.time())
print(ktc.labels_)
print(Counter(ktc.labels_))
print(np.diff(timelist))
labelmat = labels.reshape(1, -1) == labels.reshape(-1, 1)


def pairwise_counting (PC, TC):
	# Get Pair Counter Matrix
	reduced_PC = np.unique(PC, return_inverse=True)[1]
	reduced_TC = np.unique(TC, return_inverse=True)[1]
	pair_counter = np.unique(np.transpose([reduced_PC, reduced_TC]), return_counts=True, axis=0)
	pair_counter_matrix = coo_matrix((pair_counter[1], pair_counter[0].T)).toarray()
	# Get Interim Stats
	D = len(reduced_PC)
	E = np.sum(pair_counter_matrix ** 2)
	R = np.sum(np.sum(pair_counter_matrix, axis=1) ** 2)
	C = np.sum(np.sum(pair_counter_matrix, axis=0) ** 2)
	# Get Final Stats
	TP = E - D
	FN = C - E
	FP = R - E
	TN = D ** 2 - TP - FN - FP - D
	return (TN, FP, FN, TP)


L1 = []
L2 = []
plabels = np.arange(1080, dtype=int)
for i in range(1080):
	confmat = pairwise_counting(plabels, labels)
	L1.append(confmat)
	print("conf iteration:", i, confmat)
	try:
		plabels[ktc._trl.tree[i][1]] = ktc._trl.tree[i][0]
		for j in range(1080):
			current = j
			while current != plabels[current]:
				current = plabels[current]
			ending = current
			current = j
			while current != ending:
				plabels[current] = ending
				current = plabels[current]
	except Exception:
		pass
for i in range(1, 400):
	model = KMeans(i)
	model.fit(imp_dataset)
	mconfmat = pairwise_counting(model.labels_, labels)
	L2.append(mconfmat)
	print("conf iteration:", i, mconfmat)
L1 = np.transpose(L1)
tp = L1[3] / (L1[3] + L1[2])
fp = L1[1] / (L1[0] + L1[1])
L2 = np.transpose(L2+[(1,0,1,0)])
tp2 = L2[3] / (L2[3] + L2[2])
fp2 = L2[1] / (L2[0] + L2[1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve of Clustering on MPE Data Set")
p1 = plt.plot(fp, tp)
p2 = plt.plot(fp2, tp2)
auc = np.trapz(tp, fp)
auc2 = -np.trapz(tp2, fp2)
plt.legend((p1[0], p2[0]), ("KT AUC = {:.3f}".format(auc), "KMeans AUC = {:.3f}".format(auc2)), loc="lower right")
plt.show()
"""
from sklearn import metrics
L1 = []
L2 = []
plabels = np.arange(1080, dtype=int)
for i in range(1080):
	confmat = metrics.v_measure_score(labels, plabels)
	L1.append(confmat)
	print("vmeasure iteration:", i, confmat)
	try:
		plabels[ktc._trl.tree[i][1]] = ktc._trl.tree[i][0]
		for j in range(1080):
			current = j
			while current != plabels[current]:
				current = plabels[current]
			ending = current
			current = j
			while current != ending:
				plabels[current] = ending
				current = plabels[current]
	except Exception:
		pass
L1 = list(reversed(L1))
for i in range(2, 1080):
	model = KMeans(i)
	model.fit(imp_dataset)
	mconfmat = metrics.v_measure_score(labels, model.labels_)
	L2.append(mconfmat)
	print("vmeasure iteration:", i, mconfmat)
L1 = np.array(L1)
L2 = np.array(L2)
plt.xlabel("number of clusters")
plt.ylabel("V Measure")
plt.title("V Measure Scores of Clustering on MPE Data Set")
p1 = plt.plot(L1)
p2 = plt.plot(L2)
plt.legend((p1[0], p2[0]), ("KT", "KMeans"), loc="lower right")
plt.show()"""