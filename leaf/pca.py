import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('leaf.csv',sep= ',', header= None)
X = balance_data.values[:, 1:15]
Y = balance_data.values[:,0]
for i in range(1,2):
    pca=PCA(n_components=14,random_state=100)
    pca.fit(X)
    print(pca.explained_variance_)
    print(len(pca.explained_variance_))
    new_X=pca.transform(X)
    t=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    #np.savetxt('PCA'+str(i)+'.csv',new_X)
    plt.bar(t,pca.explained_variance_)
    plt.show()

