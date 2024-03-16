from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

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
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clusters, init):
    data, labels = dataset
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    estimator = KMeans(init=init, random_state=42, n_clusters=n_clusters, n_init='auto')
    estimator.fit(data)
    preds = estimator.labels_
    SSE = np.sum((preds - labels)**2)
    return SSE#, estimator.inertia_



def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    dataset = make_blobs(center_box=(-20,20), n_samples=20, centers=5, random_state=12)
    data, labels = dataset
    x_coords = data[:,0]
    y_coords = data[:,1]

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [] ##################################################

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    
    # ks = [1,2,3,4,5,6,7,8]
    # sse_list = []
    # inertia_list = []
    # for k in ks:
    #     SSE, inertia = fit_kmeans(dataset=dataset, n_clusters=k, init='random')
    #     sse_list.append([k, SSE])
    #     inertia_list.append([k, inertia])


    dct = answers["2C: SSE plot"] = [[1, 100]]#########

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = [[1,100]]###########

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = ""

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
