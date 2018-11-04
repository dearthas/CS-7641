import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('seeds.csv',sep= ',', header= None)
X = balance_data.values[:, 0:6]
Y = balance_data.values[:,7]
for i in range(1,7):
    ica=FastICA(n_components=i,random_state=100,max_iter=1000)
    ica.fit(X)
    new_X=ica.transform(X)
    np.savetxt('ICA'+str(i)+'.csv',new_X)
