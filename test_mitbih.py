import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix

from kernel_treelet_clustering import kernel_treelet_clustering

trds = np.genfromtxt(r"datasets\\mitbih_test.csv", delimiter=',')[15000:]
trlab = np.asarray(trds[:, -1], dtype=int)
print(np.unique(trlab, return_counts=True))
trds = trds[:, :-1]
n = len(trlab)
ktc = kernel_treelet_clustering('rbf', sigma=0.3, number_of_clusters=5, max_sample=9000, verbose=True)
ktc.fit(trds)


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


L = []
labels = np.arange(n, dtype=int)
for i in range(n):
	confmat = pairwise_counting(labels, trlab)
	print("conf iteration:", i, confmat)
	L.append(confmat)
	try:
		labels[ktc._trl.tree[i][1]] = ktc._trl.tree[i][0]
		for j in range(n):
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
tp = L[3] / (L[3] + L[2])
fp = L[1] / (L[0] + L[1])

plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.title("MIT-BIH ROC curve")
plt.plot(fp, tp)
auc = np.trapz(tp, fp)
plt.text(0.7, 0.1, "AUC = {:.3f}".format(auc))
plt.show()

