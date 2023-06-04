import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.csv")
print(iris.head())

versicolor_petal_length = np.array(iris[iris.variety == "Versicolor"])[:,2]
mean_length_vers = np.mean(versicolor_petal_length)
print('I. versicolor:', mean_length_vers, 'cm')

perc = np.array([2.5, 25, 50, 75, 97.5])
ptiles_vers = np.percentile (versicolor_petal_length, perc)
print(ptiles_vers)

from Iris import ecdf

x_vers, y_vers = ecdf(versicolor_petal_length)

plt.plot(x_vers, y_vers, '.')
plt.xlabel('petal length (cm)')
plt.ylabel('ECDF')

plt.plot(ptiles_vers, perc/100, marker='D', color='red', linestyle='none')
##plt.show()

sns.boxplot(x='variety', y='petal.length', data=iris)
plt.xlabel('variety')
plt.ylabel('petal length (cm)')
plt.show()

variance = np.var(versicolor_petal_length)
print(np.sqrt(variance))
print(np.std(versicolor_petal_length))

versicolor_petal_width = np.array(iris[iris.variety == "Versicolor"])[:,3]
plt.plot (versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')
plt.ylabel('petal width (cm)')
plt.xlabel('petal length (cm)')
plt.show()

covariance_matrix = np.cov(versicolor_petal_length.astype(float), versicolor_petal_width.astype(float))
print(covariance_matrix)
petal_cov = covariance_matrix[0,1]
print(petal_cov)

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor: r
r = pearson_r(versicolor_petal_length.astype(float), versicolor_petal_width.astype(float))

# Print the result
print(r)



