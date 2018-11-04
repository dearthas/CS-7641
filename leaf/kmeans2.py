from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('leaf.csv',sep= ',', header= None)
X = balance_data.values[:, 1:15]
Y = balance_data.values[:,0]
t=[]
result=[]
for i in range(0,20):
    estimator = KMeans(n_clusters=30,n_init=10,tol=1e-20)
    estimator.fit(X)
    Y_pred=estimator.predict(X)
    print("Accuracy:",metrics.adjusted_rand_score(Y, Y_pred))
    result.append(metrics.adjusted_rand_score(Y, Y_pred))
    t.append(1+i*5)

plt.plot(t,result)

plt.show()
