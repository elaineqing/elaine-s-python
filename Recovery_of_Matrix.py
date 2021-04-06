%matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

#generate sign for the matrix
sign = []
for _ in range(100):
    sign.append(np.random.choice((-1, 1))) 

#generate random value for the matrix
X = (np.random.rand(10**2)*sign).reshape(10, 10)

#generate symmetric matrix
X = np.triu(X)
X += X.T - np.diag(X.diagonal())

#print(X)
#print(X.T == X)

plt.matshow(X)


#eigenValue and eigenVector
eigValue,eigVector = la.eig(X)

#sort both by absolute value of eigenvalue
idx = abs(eigValue).argsort()[::-1]   
eigValue = eigValue[idx]
eigVector = eigVector[:,idx]

#recover for 2 values
Y = eigVector[0:1] * np.diag(eigValue)[0:1] * np.linalg.inv(eigVector)[0:1]
plt.matshow(Y.real)

#recover for 4 values
Z = eigVector[0:3] * np.diag(eigValue)[0:3] * np.linalg.inv(eigVector)[0:3]
plt.matshow(Z.real)

plt.show()
