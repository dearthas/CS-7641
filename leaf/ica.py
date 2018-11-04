import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('leaf.csv',sep= ',', header= None)
X = balance_data.values[:, 1:15]
Y = balance_data.values[:,0]
for i in range(1,16):
    pca=FastICA(n_components=i,random_state=100)
    pca.fit(X)
    new_X=pca.transform(X)
    np.savetxt('ICA'+str(i)+'.csv',new_X)
