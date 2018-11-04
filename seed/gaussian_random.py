import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('seeds.csv',sep= ',', header= None)
X = balance_data.values[:, 0:6]
Y = balance_data.values[:,7]
for i in range(1,7):
    gaussian=GaussianRandomProjection(n_components=6)
    new_X=gaussian.fit_transform(X)
    np.savetxt('GAUSSIAN'+str(i)+'.csv',new_X)
