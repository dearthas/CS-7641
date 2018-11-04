from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('seeds.csv',sep= ',', header= None)
X = balance_data.values[:, 0:6]
Y = balance_data.values[:,7]
estimator = KMeans(n_clusters=3)
estimator.fit(X)
Y_pred=estimator.predict(X)
print("Accuracy:",metrics.adjusted_rand_score(Y, Y_pred))
np.savetxt('kmeans.csv',Y_pred)
