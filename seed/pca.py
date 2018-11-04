import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('seeds.csv',sep= ',', header= None)
X = balance_data.values[:, 0:6]
Y = balance_data.values[:,7]
for i in range(1,2):
    pca=PCA(n_components=6,random_state=100)
    pca.fit(X)
    new_X=pca.transform(X)
    t=[1,2,3,4,5,6]
    print(pca.explained_variance_)
    #np.savetxt('PCA'+str(i)+'.csv',new_X)
    plt.bar(t,pca.explained_variance_)
    plt.show()
