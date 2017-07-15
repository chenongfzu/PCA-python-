import os
import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(1, 11)
y = 2 * x + np.random.randn(10) * 2
X = np.vstack((x, y))
print("x=", x)
print("y=", y)
print("X=", X)
fig = plt.figure()
scatter1 = plt.scatter(x, y)
grid1 = plt.grid(True)
print('Scatter: ', type(scatter1))
plt.title("Arrange list")
# plt.show()
Xcentered = (X[0] - x.mean(), X[1] - y.mean())
m = (x.mean(), y.mean())
print(Xcentered)
print("Mean vector: ", m)
fig2 = plt.figure()
scatter1 = plt.scatter(X[0] - x.mean(), X[1] - y.mean())
grid2 = plt.grid(True)
print('Scatter: ', type(scatter1))
plt.title("Mean values")
covmat = np.cov(Xcentered)
print("covmat=", covmat, "\n")
print("Variance of X: ", np.cov(Xcentered)[0, 0])
print("Variance of Y: ", np.cov(Xcentered)[1, 1])
print("Covariance X and Y: ", np.cov(Xcentered)[0, 1])
_, vecs = np.linalg.eig(covmat)
v = -vecs[:, 1]
Xnew = np.dot(v, Xcentered)
print("Xnew=", Xnew)
plt.show()