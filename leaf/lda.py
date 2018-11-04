import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('leaf.csv',sep= ',', header= None)
X = balance_data.values[:, 1:15]
Y = balance_data.values[:,0]
for i in range(1,30):
    lda=LinearDiscriminantAnalysis(n_components =i)
    lda.fit(X,Y)
    new_X=lda.fit_transform(X,Y)
    np.savetxt('LDA'+str(i)+'.csv',new_X)
