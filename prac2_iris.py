from sklearn.datasets import load_iris
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import sys


iris = load_iris()
X, y = iris['data'], iris['target']
X = np.array(X)
y = np.array(y)
N,D = X.shape
Ntrain = int(0.8*N)
K = 500
k = 10
err_nbc = np.zeros(k)
err_lr = np.zeros(k)

for j in range(K):
    shuffler = np.random.permutation(N)
    Xtrain = X[shuffler[:Ntrain]]
    ytrain = y[shuffler[:Ntrain]]
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]
    i = 1
    
    while i < k + 1:
        lr = LogisticRegression(solver='newton-cg', multi_class='multinomial')
        nbc = GaussianNB()
        lr_p = lr.fit(Xtrain[:int(0.1*i*Ntrain)], ytrain[:int(0.1*i*Ntrain)]).predict(Xtest)
        nbc_p = nbc.fit(Xtrain[:int(0.1*i*Ntrain)], ytrain[:int(0.1*i*Ntrain)]).predict(Xtest)
        err_nbc[i-1] += np.mean(nbc_p == ytest)
        err_lr[i-1] += np.mean(lr_p == ytest)
        i += 1
        
err_nbc, err_lr = err_nbc/K, err_lr/K
z = np.arange(k)*0.1 + 0.1
plt.plot(z, err_nbc, 'r', z, err_lr, 'b')
plt.show()