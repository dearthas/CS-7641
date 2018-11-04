import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
balance_data = pd.read_csv('em_seeds.csv',sep= ',', header= None)
X = balance_data.values[:, 0:7]
Y = balance_data.values[:,8]
t=[]
result=[]
for i in range(1,20):
    start= time.time()
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(60),max_iter=900, random_state=0,tol=1e-10,activation='tanh',learning_rate_init=0.00001,learning_rate='invscaling')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    end=time.time()
    print("time :",end-start)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t.append(i*5)
    result.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(t,result)

plt.show()
