from sklearn.mixture import GMM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('seeds.csv',sep= ',', header= None)
X = balance_data.values[:, 0:6]
Y = balance_data.values[:,7]
t=[]
result=[]
for i in range(1,10):
    estimator = GMM(n_components=i,covariance_type='full')
    estimator.fit(X)
    Y_pred=estimator.predict(X)
    print("Accuracy:",metrics.adjusted_rand_score(Y, Y_pred))
    result.append(metrics.adjusted_rand_score(Y, Y_pred))
    t.append(i)

plt.plot(t,result)

plt.show()
