from sklearn.mixture import GMM
import numpy as np
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('leaf.csv',sep= ',', header= None)
X = balance_data.values[:, 1:15]
Y = balance_data.values[:,0]
estimator = GMM(n_components=30,n_iter=16,covariance_type='diag')
estimator.fit(X)
Y_pred=estimator.predict(X)
print("Accuracy:",metrics.adjusted_rand_score(Y, Y_pred))
np.savetxt('em.csv',Y_pred)
