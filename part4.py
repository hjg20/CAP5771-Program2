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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(dataset, linkage_type, n_clusters):
    data, labels = dataset
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    estimator = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    estimator.fit(data)
    return estimator.labels_

def fit_modified(dataset, linkage_type):
    data, labels = dataset
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    Z = linkage(data, method=linkage_type)

    diff = np.diff(Z[:, 2])
    max_diff_index = np.argmax(diff)
    cutoff_distance = Z[max_diff_index, 2]

    labels = fcluster(Z, cutoff_distance, criterion='distance')
    
    return labels


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
    SAMPLES = 100
    SEED = 42
    FACTOR = 0.5
    NOISE = 0.05
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

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {
        "nc": [nc_data, nc_labels],
        "nm": [nm_data, nm_labels],
        "bvv": [bvv_data, bvv_labels],
        "add": [add_data, add_labels],
        "b": [b_data, b_labels]}

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations (see 1.C)
    fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))

    nc_pred_labels = fit_hierarchical_cluster((nc_data, nc_labels), n_clusters=2, linkage_type='single')
    nm_pred_labels = fit_hierarchical_cluster((nm_data, nm_labels), n_clusters=2, linkage_type='single')
    bvv_pred_labels = fit_hierarchical_cluster((bvv_data, bvv_labels), n_clusters=2, linkage_type='single')
    add_pred_labels = fit_hierarchical_cluster((add_data, add_labels), n_clusters=2, linkage_type='single')
    b_pred_labels = fit_hierarchical_cluster((b_data, b_labels), n_clusters=2, linkage_type='single')

    nc_pred_labels2 = fit_hierarchical_cluster((nc_data, nc_labels), n_clusters=2, linkage_type='complete')
    nm_pred_labels2 = fit_hierarchical_cluster((nm_data, nm_labels), n_clusters=2, linkage_type='complete')
    bvv_pred_labels2 = fit_hierarchical_cluster((bvv_data, bvv_labels), n_clusters=2, linkage_type='complete')
    add_pred_labels2 = fit_hierarchical_cluster((add_data, add_labels), n_clusters=2, linkage_type='complete')
    b_pred_labels2 = fit_hierarchical_cluster((b_data, b_labels), n_clusters=2, linkage_type='complete')

    nc_pred_labels3 = fit_hierarchical_cluster((nc_data, nc_labels), n_clusters=2, linkage_type='ward')
    nm_pred_labels3 = fit_hierarchical_cluster((nm_data, nm_labels), n_clusters=2, linkage_type='ward')
    bvv_pred_labels3 = fit_hierarchical_cluster((bvv_data, bvv_labels), n_clusters=2, linkage_type='ward')
    add_pred_labels3 = fit_hierarchical_cluster((add_data, add_labels), n_clusters=2, linkage_type='ward')
    b_pred_labels3 = fit_hierarchical_cluster((b_data, b_labels), n_clusters=2, linkage_type='ward')

    nc_pred_labels4 = fit_hierarchical_cluster((nc_data, nc_labels), n_clusters=2, linkage_type='average')
    nm_pred_labels4 = fit_hierarchical_cluster((nm_data, nm_labels), n_clusters=2, linkage_type='average')
    bvv_pred_labels4 = fit_hierarchical_cluster((bvv_data, bvv_labels), n_clusters=2, linkage_type='average')
    add_pred_labels4 = fit_hierarchical_cluster((add_data, add_labels), n_clusters=2, linkage_type='average')
    b_pred_labels4 = fit_hierarchical_cluster((b_data, b_labels), n_clusters=2, linkage_type='average')

    axs[0, 0].scatter(nc_data[:, 0], nc_data[:, 1], c=nc_pred_labels)
    axs[0, 1].scatter(nc_data[:, 0], nc_data[:, 1], c=nc_pred_labels2)
    axs[0, 2].scatter(nc_data[:, 0], nc_data[:, 1], c=nc_pred_labels3)
    axs[0, 3].scatter(nc_data[:, 0], nc_data[:, 1], c=nc_pred_labels4)

    axs[1, 0].scatter(nm_data[:, 0], nm_data[:, 1], c=nm_pred_labels)
    axs[1, 1].scatter(nm_data[:, 0], nm_data[:, 1], c=nm_pred_labels2)
    axs[1, 2].scatter(nm_data[:, 0], nm_data[:, 1], c=nm_pred_labels3)
    axs[1, 3].scatter(nm_data[:, 0], nm_data[:, 1], c=nm_pred_labels4)

    axs[2, 0].scatter(bvv_data[:, 0], bvv_data[:, 1], c=bvv_pred_labels)
    axs[2, 1].scatter(bvv_data[:, 0], bvv_data[:, 1], c=bvv_pred_labels2)
    axs[2, 2].scatter(bvv_data[:, 0], bvv_data[:, 1], c=bvv_pred_labels3)
    axs[2, 3].scatter(bvv_data[:, 0], bvv_data[:, 1], c=bvv_pred_labels4)

    axs[3, 0].scatter(add_data[:, 0], add_data[:, 1], c=add_pred_labels)
    axs[3, 1].scatter(add_data[:, 0], add_data[:, 1], c=add_pred_labels2)
    axs[3, 2].scatter(add_data[:, 0], add_data[:, 1], c=add_pred_labels3)
    axs[3, 3].scatter(add_data[:, 0], add_data[:, 1], c=add_pred_labels4)

    axs[4, 0].scatter(b_data[:, 0], b_data[:, 1], c=b_pred_labels)
    axs[4, 1].scatter(b_data[:, 0], b_data[:, 1], c=b_pred_labels2)
    axs[4, 2].scatter(b_data[:, 0], b_data[:, 1], c=b_pred_labels3)
    axs[4, 3].scatter(b_data[:, 0], b_data[:, 1], c=b_pred_labels4)

    dct = answers["4B: cluster successes"] = ["nc", "nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))

    nc_pred_labels = fit_modified((nc_data, nc_labels), linkage_type='single')
    nm_pred_labels = fit_modified((nm_data, nm_labels), linkage_type='single')
    bvv_pred_labels = fit_modified((bvv_data, bvv_labels), linkage_type='single')
    add_pred_labels = fit_modified((add_data, add_labels), linkage_type='single')
    b_pred_labels = fit_modified((b_data, b_labels), linkage_type='single')

    nc_pred_labels2 = fit_modified((nc_data, nc_labels), linkage_type='complete')
    nm_pred_labels2 = fit_modified((nm_data, nm_labels), linkage_type='complete')
    bvv_pred_labels2 = fit_modified((bvv_data, bvv_labels), linkage_type='complete')
    add_pred_labels2 = fit_modified((add_data, add_labels), linkage_type='complete')
    b_pred_labels2 = fit_modified((b_data, b_labels), linkage_type='complete')

    nc_pred_labels3 = fit_modified((nc_data, nc_labels), linkage_type='ward')
    nm_pred_labels3 = fit_modified((nm_data, nm_labels), linkage_type='ward')
    bvv_pred_labels3 = fit_modified((bvv_data, bvv_labels), linkage_type='ward')
    add_pred_labels3 = fit_modified((add_data, add_labels), linkage_type='ward')
    b_pred_labels3 = fit_modified((b_data, b_labels), linkage_type='ward')

    nc_pred_labels4 = fit_modified((nc_data, nc_labels), linkage_type='average')
    nm_pred_labels4 = fit_modified((nm_data, nm_labels), linkage_type='average')
    bvv_pred_labels4 = fit_modified((bvv_data, bvv_labels), linkage_type='average')
    add_pred_labels4 = fit_modified((add_data, add_labels), linkage_type='average')
    b_pred_labels4 = fit_modified((b_data, b_labels), linkage_type='average')

    axs[0, 0].scatter(nc_data[:, 0], nc_data[:, 1], c=nc_pred_labels)
    axs[0, 1].scatter(nc_data[:, 0], nc_data[:, 1], c=nc_pred_labels2)
    axs[0, 2].scatter(nc_data[:, 0], nc_data[:, 1], c=nc_pred_labels3)
    axs[0, 3].scatter(nc_data[:, 0], nc_data[:, 1], c=nc_pred_labels4)

    axs[1, 0].scatter(nm_data[:, 0], nm_data[:, 1], c=nm_pred_labels)
    axs[1, 1].scatter(nm_data[:, 0], nm_data[:, 1], c=nm_pred_labels2)
    axs[1, 2].scatter(nm_data[:, 0], nm_data[:, 1], c=nm_pred_labels3)
    axs[1, 3].scatter(nm_data[:, 0], nm_data[:, 1], c=nm_pred_labels4)

    axs[2, 0].scatter(bvv_data[:, 0], bvv_data[:, 1], c=bvv_pred_labels)
    axs[2, 1].scatter(bvv_data[:, 0], bvv_data[:, 1], c=bvv_pred_labels2)
    axs[2, 2].scatter(bvv_data[:, 0], bvv_data[:, 1], c=bvv_pred_labels3)
    axs[2, 3].scatter(bvv_data[:, 0], bvv_data[:, 1], c=bvv_pred_labels4)

    axs[3, 0].scatter(add_data[:, 0], add_data[:, 1], c=add_pred_labels)
    axs[3, 1].scatter(add_data[:, 0], add_data[:, 1], c=add_pred_labels2)
    axs[3, 2].scatter(add_data[:, 0], add_data[:, 1], c=add_pred_labels3)
    axs[3, 3].scatter(add_data[:, 0], add_data[:, 1], c=add_pred_labels4)

    axs[4, 0].scatter(b_data[:, 0], b_data[:, 1], c=b_pred_labels)
    axs[4, 1].scatter(b_data[:, 0], b_data[:, 1], c=b_pred_labels2)
    axs[4, 2].scatter(b_data[:, 0], b_data[:, 1], c=b_pred_labels3)
    axs[4, 3].scatter(b_data[:, 0], b_data[:, 1], c=b_pred_labels4)

    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
