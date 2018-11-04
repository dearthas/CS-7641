import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('leaf.csv',sep= ',', header= None)
X = balance_data.values[:, 1:15]
Y = balance_data.values[:,0]
for i in range(1,15):
    gaussian=GaussianRandomProjection(n_components=14)
    new_X=gaussian.fit_transform(X)
    np.savetxt('GAUSSIAN'+str(i)+'.csv',new_X)
