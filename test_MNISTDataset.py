import collections
import os
import struct
import time

import numpy as np
from sklearn.svm import LinearSVC

from kernel_treelet_clustering import kernel_treelet_clustering

timelist = []
datadir = r"C:\D\senior_thesis\handwritten_num\samples"


def read_idx (*filenames):
	output = []
	for filename in filenames:
		with open(os.path.join(datadir, filename), 'rb') as f:
			zero, data_type, dims = struct.unpack('>HBB', f.read(4))
			shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
			output.append(np.fromstring(f.read(), dtype=np.uint8).reshape(shape))
	return tuple(output)


accuracylist = []

train_x, train_y, test_x, test_y = read_idx("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
train_x = np.asarray(train_x, dtype=np.float_).reshape(-1, 28 * 28) / 256
test_x = np.asarray(test_x, dtype=np.float_).reshape(-1, 28 * 28) / 256


supervised_method = LinearSVC
print("start training")
timelist.append(time.time())

iterations = 10
for i in range(iterations):
	print("iteration:", i)
	ktc = kernel_treelet_clustering('rbf', 30, 500, sigma=1, verbose=True)
	#ktc = kernel_treelet_clustering('poly', 30, 500, _gamma_=1, coef0=5, degree=10, verbose=True)
	print(ktc.kernel_name)
	pred = np.zeros(test_y.shape)
	if ktc.kernel_name is None:
		supervised_model = supervised_method()
		supervised_model.fit(train_x, train_y)
		pred += supervised_model.predict(test_x)
	else:
		ktc.fit(train_x)
		print(len(set(ktc.labels_)), collections.Counter(ktc.labels_))
		cluster_assignment = ktc.svm.predict(test_x)
		labels_ = ktc.labels_
		for cluster in set(ktc.labels_):
			if len(set(train_y[labels_ == cluster])) == 1:
				pred += np.full(test_y.shape, cluster, dtype=int) * (cluster_assignment == cluster)
			else:
				supervised_model = supervised_method()
				supervised_model.fit(train_x[labels_ == cluster], train_y[labels_ == cluster])
				pred += supervised_model.predict(test_x) * (cluster_assignment == cluster)
	accuracy = np.mean(pred == test_y)
	print(accuracy)
	accuracylist.append(accuracy)
	timelist.append(time.time())

print(np.diff(timelist))
print(np.mean(np.diff(timelist)))
print(np.mean(accuracylist), np.std(accuracylist))


# Record doc

record_path = r"C:\Users\Hedi Xia\Desktop\temp\record.csv"

# CLF, K, n, t, a = str(supervised_method), ktc.kernel_name, ktc.max_sample, np.diff(timelist), accuracylist

record_bin = open(record_path, "ab")
record = np.repeat([supervised_method.__name__, ktc.kernel_name, ktc.max_sample], iterations).reshape(3, -1)
record = np.transpose(np.concatenate([record, [np.diff(timelist), accuracylist]]))
print(record)
np.savetxt(record_bin, record, delimiter=",", fmt="%s")


