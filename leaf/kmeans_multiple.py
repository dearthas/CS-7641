from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
balance_data = pd.read_csv('leaf.csv',sep= ',', header= None)
t=[]
result=[]
color=['r','b','y','black']
k=0
fig, ax = plt.subplots()
for e in ['PCA','ICA','LDA','GAUSSIAN']:
    t=[]
    result=[]
    for i in range(1,15):
        balance_data2 = pd.read_csv(e+"/"+e+str(i)+'.csv',sep= ',', header= None)
        X = balance_data2.values[:, 0:14]
        Y = balance_data.values[:,0]
        estimator = KMeans(n_clusters=30)
        estimator.fit(X)
        Y_pred=estimator.predict(X)
        print("Accuracy:",metrics.adjusted_rand_score(Y, Y_pred))
        t.append(i)
        result.append(metrics.adjusted_rand_score(Y, Y_pred))

    ax.plot(t,result,color[k],label=e)
    k=k+1

legend = ax.legend(loc=(0.8,0.1), shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')


plt.show()
