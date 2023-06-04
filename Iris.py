import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.csv")
print(iris.head())
iris_versicolor = iris[iris.variety == 'Versicolor']
petal_length = np.array(iris_versicolor)[:,2]

sns.set()
##plt.hist(petal_length)
##plt.xlabel('petal length (cm)')
##plt.ylabel('count')
##plt.show()

##ndata = len(petal_length)
##nbins = int(np.sqrt(ndata))
##sns.set()
##plt.hist(petal_length, bins = nbins )
##plt.show()
##plt.xlabel('petal length (cm)')
##plt.ylabel('count')
##
##sns.swarmplot(x='variety', y='petal.length', data=iris, s=3)
##plt.xlabel('species')
##plt.ylabel('petal length (cm)')
##plt.show()

##n=len(iris_versicolor)

##def ecdf(n, column):
##    
##    print(n)
##    x=iris.sort_values(by=column)
####x=np.sort(iris_versicolor)
##    y=np.arange(1, n+1)/n
##    print(y)

def ecdf(data):

    n=len(data)
    print(n)
    x=np.sort(data)
    y=np.arange(1, n+1)/n
    return x, y

##cdf, pet_length = ecdf(petal_length)
##plt.plot(pet_length, cdf, marker='.', linestyle='none')
##plt.ylabel('ecdf)')
##plt.xlabel('petal length (cm)')
##plt.show()

vers_petal = np.array(iris[iris.variety == "Versicolor"])[:,2]
set_petal = np.array(iris[iris.variety == "Setosa"])[:,2]
vir_petal = np.array(iris[iris.variety == "Virginica"])[:,2]

x_vers, y_vers = ecdf(vers_petal)
x_set, y_set = ecdf(set_petal)
x_virg, y_virg = ecdf(vir_petal)

##plt.plot(x_vers, y_vers, marker='.', linestyle='none', color='blue')
##plt.plot(x_set, y_set, marker='x', linestyle='none', color='red')
##plt.plot(x_virg, y_virg, marker='o', linestyle='none')
##plt.show()
