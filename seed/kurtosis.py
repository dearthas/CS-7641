import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import pandas as pd
balance_data = pd.read_csv('ICA/ICA6.csv',sep= ',', header= None)
t=[]
result=[]
for i in range(0,6):
    X=balance_data.values[:,i]
    result.append(kurtosis(X))
    print(kurtosis(X))
    t.append(i+1)
result.sort()
result.reverse()
plt.bar(t,result)

plt.show()
