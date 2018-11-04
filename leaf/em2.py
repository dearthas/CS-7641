from sklearn.mixture import GMM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('leaf.csv',sep= ',', header= None)
X = balance_data.values[:, 1:15]
Y = balance_data.values[:,0]
t=[]
result=[]
for i in range(1,25):
    estimator = GMM(n_components=30,covariance_type='diag',tol=1e-20,random_state=100,n_iter=16)
    estimator.fit(X)
    Y_pred=estimator.predict(X)
    print("Accuracy:",metrics.adjusted_rand_score(Y, Y_pred))
    result.append(metrics.adjusted_rand_score(Y, Y_pred))
    t.append(i)

plt.plot(t,result)

plt.show()
