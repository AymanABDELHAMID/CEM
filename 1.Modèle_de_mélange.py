"""
Ayman Mahmoud
DAML

Exercice 1 : (Simulation d’un modèle de mélange) (Modèle de mélange Gaussien = Gaussian Mixture Model) (CH3 - Slide 43)
    — Écrire une fonction qui simule 'n' points du plan suivant un modèle de mélange de deux lois gaussiennes
        définies par leurs proportions, centres, volumes, formes et orientations.
    — Donner quelques exemples de simulation et donner les représentations graphiques correspondantes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

## GMM
# proportions, centres, volumes, formes et orientations. (Slide 45 - CH3) (to estimate parameters)
# Modèles probabilistes en Classification Part 1: exemple de mélange - slide 21 (one dimension)
# Mémoire Thématique (Dauphine) - Page 10 - pour deux dimensions


# 1. Reading Data
data = pd.read_csv(
    '../../OneDrive - CentraleSupelec/CentraleSupelec/M2_MACLO/DAML/Livrable_Cours_DAML/Projet Classification/Livrable_Projet_AM_AMB/Code/data/exo1/Clustering_gmm.csv')

plt.figure(figsize=(7,7))
plt.scatter(data["Weight"],data["Height"])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Data Distribution')
#plt.show()
plt.savefig('./output/data.png')

# Transformer le dataframe en array
data_array = data.values

# Nombre des lois Gaussiennes
k = 2

# Create the array r with dimensionality nxK (n = number of datapoints, k = number of gaussians)
r = np.zeros((len(data_array),2)) # for exact means and covariances
r_2 = np.zeros((len(data_array),2)) #for estimated means and covariances

print('Dimensionality','=',np.shape(r))

# Créer les Gaussiennes (estimate the values)
# Using: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

# Need to use sklearn to estimate the paramaters (mu, covariance)
gmm = GaussianMixture(n_components=2)
gmm.fit(data_array)

# GMM Parameters
Z = gmm.covariances_[0]
mu = gmm.means_[0]

Z2 = gmm.covariances_[1]
mu2 = gmm.means_[1]

# Estimated Parameters
Z_0  = [[ 1.5 ,  1],
        [ 1,  1]]
mu_0 = [54, 160]

Z_0_2 = [[ 9,  10],
        [ 8, 10 ]]
mu_0_2 = [60, 170]

gauss_1_1 = multivariate_normal(mu,Z)
gauss_1_2 = multivariate_normal(mu_0,Z_0)
gauss_2_1 = multivariate_normal(mu2,Z2)
gauss_2_2 = multivariate_normal(mu_0_2,Z_0_2)

# in order to estimate (pi) it is the percentage of the number of points you expect to be in that cluster
pi = np.array([0.5,0.5])

# Write the probability that x belongs to gaussian c in column c.
for c,g in zip(range(3),[gauss_1_1,gauss_2_1]):
    r[:,c] = g.pdf(data_array) # (remember that pdf returns 4 values)

# Write the probability that x belongs to gaussian c in column c.
for c,g in zip(range(3),[gauss_1_1,gauss_2_1]):
    r_2[:,c] = g.pdf(data_array) # (remember that pdf returns 4 values)

# normalizing probabilities
for i in range(len(r)):
   r[i] = r[i]/(np.sum(pi)*np.sum(r,axis=1)[i])
   r_2[i] = r_2[i]/(np.sum(pi)*np.sum(r,axis=1)[i])
# for testing
# np.sum(r,axis=1)

r2_1 = np.zeros((len(data_array),1))
r2_2 = np.zeros((len(data_array),1))
for i in range(len(r)):
    if r[i,0] > r[i,1]:
        r2_1[i] = 1
    else:
        r2_1[i] = 2

    if r_2[i,0] > r_2[i,1]:
        r2_2[i] = 1
    else:
        r2_2[i] = 2

frame = pd.DataFrame(data)
frame['cluster'] = r2_1
frame['cluster2'] = r2_2
frame.columns = ['Weight', 'Height', 'cluster', 'cluster2']

# plotting data:
color=['blue','green']
for k in range(1,3):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k-1])
#plt.show()
plt.savefig('./output/data_exact.png')


# plotting data:
color=['blue','green']
for k in range(1,3):
    data = frame[frame["cluster2"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k-1])
#plt.show()
plt.savefig('./output/data_estimate.png')

# look for how to estimate parameters