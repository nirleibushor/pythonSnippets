import numpy as np
import matplotlib.pyplot as plt

"""
create random 2-dimensional data, separated by different radius ranges
"""
N = 400
D = 2
data = np.ndarray((N, D))

theta = np.random.uniform(low=0.0, high=2 * np.pi, size=N)
smallRs = np.random.uniform(low=0.0, high=1.0, size=N / 2)
largeRs = np.random.uniform(low=2.0, high=3.0, size=N / 2)
r = np.concatenate((smallRs, largeRs))

for i in xrange(N):
    data[i, 0] = np.cos(theta[i]) * r[i]
    data[i, 1] = np.sin(theta[i]) * r[i]

labels = ['blue' if (np.sqrt(data[i, 0] ** 2 + data[i, 1] ** 2) <= 1.5) else 'red' for i in xrange(N)]

plt.scatter(data[:, 0], data[:, 1], color=labels)
plt.show()

ndarray_labels = np.ndarray((N,))

for i in xrange(N):
    ndarray_labels[i] = 1 if labels == 'red' else 0

print 'data shape: ', data.shape
print 'labels shape:', ndarray_labels.shape

i = 150
print 'Point number', i, ':', '(', data[i, 0], ',', data[i, 1], ')', labels[i]