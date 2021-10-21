"""
Ayman Mahmoud
DAML

Exercice 2:
    Programmer l’algorithme CEM pour les modèles parcimonieux [pi; lamdbaI],
    [pik; lamdbaI], [pi; lamdbakI] et [pik; lamdbakI]. (Slides 48, 49, 50)
"""
from utils import plotter
from matplotlib import style
from sklearn.datasets import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
import math as m
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd


# Data generation is shown in:

#############################
# 0. Create dataset
#############################

num_points = 600
#centers = [(-5, -5, -5), (0, 0, 0), (5, 5, 5)]
#X, Z = make_blobs(n_samples=num_points, n_features=3, centers=centers, shuffle=False,
#                  random_state=42)

X,z = make_classification(n_samples=num_points, n_features=3, n_informative=3,
                    n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=2,
                          class_sep=1.5,
                   flip_y=0,weights=[0.5,0.5,0.5])

"""
Implementation of the "E" step of the CEM algorithm 
"""
def expectation(X, pi, m, S):
    n, d = X.shape
    k = pi.shape[0]

    # create an array w/ kxn dimensionality
    g_arr = np.zeros((k, n))

    # probability for each datapoint to belong to
    # gaussian g
    for i in range(k):
        # for each dataset, centroid means, cov matrix
        #  P contains all the pdf values
        # Calculate the PDF of each distribution
        S_det = np.linalg.det(S[i])
        S_inv = np.linalg.inv(S[i])
        N = np.sqrt((2 * np.pi) ** d * S_det)

        fac = np.einsum('...k,kl,...l->...', X - m[i], S_inv, X - m[i])
        P = np.exp(-fac / 2) / N

        P[np.where(P < 1e-300)] = 1e-300


        # g_arr contains priors and pdf values
        g_arr[i] = pi[i] * P

    gSum = np.sum(g_arr, axis=0)
    g = g_arr / gSum

    # total log likelihood

    L = np.sum(np.log(gSum))

    return (g, L)

"""
Implementation of the "M" step of the CEM algorithm 
"""
def maximization(X, g, Parsimonious_model=1):
    n, d = X.shape
    k, _ = g.shape
    pi = g.sum(axis=1) / n
    m = np.dot(g, X) / g.sum(1)[:, None]
    S = np.zeros([k, d, d])

    if Parsimonious_model == 1: # [pi, λI]
        for i in range(k):
            ys = X - m[i, :]
            S[i] = (g[i, :, None, None] *
                    np.matmul(ys[:, :, None], ys[:, None, :])).sum(axis=0)
        S = S / g.sum(axis=1)[:, None, None]
        return (pi, m, S)
    else: # [pi, λ_kI]
        for i in range(k):
            ys = X - m[i, :]
            S[i] = (g[i, :, None, None] *
                    np.matmul(ys[:, :, None], ys[:, None, :])).sum(axis=0)
        S = S / g.sum(axis=1)[:, None, None]
        return (pi, m, S)


"""
Implementation of the "C" step of the CEM algorithm
"""
def classification(g, pi, X):
    """
    Each datapoint in X will have one partition z that is based on the maximum value
    of the posterior probability
    :param g: g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
    for each data point in each cluster
    :return: Z_i which is the classification of each datapoint to the most likely
    cluster based on (t_ik)
    """
    # create an array w/ kxn dimensionality
    n, d = X.shape
    k = pi.shape[0]
    z = np.zeros((n))

    for i in range(n):
        t_ik = g[:, i]
        z[i] = t_ik.argmax()
        #for cl_0 in range(k):
        #    if cl_0 != cl:
        #        # set other cluster to 0
        #        z[i, cl_0] = 0

    return z

"""
Implementation of the CEM algorithm
"""
def CEM(X, k, z=[1000], iterations=1000, tol=1e-5):
    # Initialization
    i = 0
    l_prev = 0
    n, d = X.shape
    #m = np.zeros((k, d))
    if z[0] == 1000:
        # use the function K-means to get cluster centers
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        m = kmeans.cluster_centers_
    else:
        sum = np.zeros((k, d))
        counter = np.zeros((k))
        # calculate centroids based on the function (mu^) in the lecture slide (49)
        for j in range(k):
            for i in range(n):
                # for each cluster find the mean value, which is the centroid
                if z[i] == j:
                    counter[j] += 1
                    sum[j] += X[i]
        m = sum/counter

    pi = np.repeat((1 / k,), k)
    cov = np.tile(np.identity(d), (k, 1))
    cov = np.reshape(cov, (k, d, d))
    mean = m

    # Run expectation before going into the loop, to get the initial Gaussians and L
    # E - Step
    g, L = expectation(X, pi, mean, cov)

    while i < iterations:
        if (np.abs(l_prev - L)) <= tol:
            break
        l_prev = L

        # M - Step
        pi, mean, cov = maximization(X, g, 1)
        g, L = expectation(X, pi, mean, cov)

        # C - Step
        z_cem = classification(g, pi, X)

        i += 1

    return pi, mean, cov, g, L, z_cem

# defining the number of clusters
k = 3

pi, mean, cov, g, L, z_cem = CEM(X, k, z, iterations=1000, tol=1e-5)

# Visualizing Data - comparison
X1 = pd.DataFrame(X)
z1 = pd.Series(z)
plotter.visualize_3d(X1,z1)



z2 = pd.Series(z_cem)
plotter.visualize_3d(X1,z2)

# Calculate error:
def evaluate_error(z1, z2):
    """
    # cette methode ne marche pas car le numero de cluster pourra changer
    :param z1: actual clustering
    :param z2: clustering done by CEM algorithm
    :return: error in percentage
    """
    err = 0
    n = len(z1)
    for i in range(n):
        if z2[i] != z1[i]:
            err += 1
    return (err/n)*100

def evaluate_error_2(z1, z2, k):
    n = len(z1)
    n_k1 = np.zeros((k))
    n_k2 = np.zeros((k))

    for j in range(k):
        for i in range(n):
            if z2[i] == j:
                n_k2[j] += 1
            if z1[i] == j:
                n_k1[j] += 1

    n_k1.sort()
    n_k2.sort()

    error_cluster = (abs(n_k2-n_k1)/n_k1)*100

    avg_err = np.mean(error_cluster)
    return error_cluster, avg_err

err1, err_avg = evaluate_error_2(z1, z2, k)

print("Error: {}".format(err_avg))