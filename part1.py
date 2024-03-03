import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering, KMeans
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked.


def fit_kmeans(dataset, n_clusters):
    data, labels = dataset
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    estimator = KMeans(init='random', random_state=42, n_clusters=n_clusters)
    estimator.fit(data)
    return estimator.labels_


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """
    SAMPLES = 100
    SEED = 42
    FACTOR = 0.5
    NOISE = 0.05
    # All datasets from https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
    nc_data, nc_labels = datasets.make_circles(n_samples=SAMPLES, factor=FACTOR, noise=NOISE, random_state=SEED)
    nm_data, nm_labels = datasets.make_moons(n_samples=SAMPLES, noise=NOISE, random_state=SEED)
    bvv_data, bvv_labels = datasets.make_blobs(n_samples=SAMPLES, cluster_std=[1.0, 2.5, 0.5], random_state=SEED)
    X, y = datasets.make_blobs(n_samples=SAMPLES, random_state=SEED)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add_data, add_labels = (X_aniso, y)
    b_data, b_labels = datasets.make_blobs(n_samples=SAMPLES, random_state=SEED)

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["1A: datasets"] = {
        "nc": [nc_data, nc_labels],
        "nm": [nm_data, nm_labels],
        "bvv": [bvv_data, bvv_labels],
        "add": [add_data, add_labels],
        "b": [b_data, b_labels]
    }

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(20, 16))
    k = [2,3,5,10]
    for i in range(len(k)):
        nc_pred_labels = fit_kmeans((nc_data, nc_labels), n_clusters=k[i])
        nm_pred_labels = fit_kmeans((nm_data, nm_labels), n_clusters=k[i])
        bvv_pred_labels = fit_kmeans((bvv_data, bvv_labels), n_clusters=k[i])
        add_pred_labels = fit_kmeans((add_data, add_labels), n_clusters=k[i])
        b_pred_labels = fit_kmeans((b_data, b_labels), n_clusters=k[i])

        axs[i, 0].scatter(nc_data[:, 0], nc_data[:, 1], c=nc_pred_labels)
        axs[i, 1].scatter(nm_data[:, 0], nm_data[:, 1], c=nm_pred_labels)
        axs[i, 2].scatter(bvv_data[:, 0], bvv_data[:, 1], c=bvv_pred_labels)
        axs[i, 3].scatter(add_data[:, 0], add_data[:, 1], c=add_pred_labels)
        axs[i, 4].scatter(b_data[:, 0], b_data[:, 1], c=b_pred_labels)

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    dct = answers["1C: cluster successes"] = {"b": [3]}

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ["nc", "nm", "bvv", "add"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = [""]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()
    #print(answers['1A: datasets'])

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
