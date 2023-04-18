#This module contains methods for data Analysis

import numpy as np

def SVD(X):
    """
    This method Whitens data by doing Singular Value Decomposition.

    Parameters:
    -----------
    X: np.ndarray
        2d array with (objects, properties)

    Returns:
    --------
    Xw: np.ndarray
        Data from SVD eigenvectors
    P: np.ndarray
        Covariance of properties
    """
    R = np.dot(X.T,X)
    U,S,V = np.linalg.svd(R)
    P = np.dot(np.diag(1./np.sqrt(S)),U.T)
    Xw = np.dot(X,P.T)

    return Xw,P

def PCA(X):
    """
    This method applies a Principal Component Analysis on the data.

    Parameters:
    -----------
    X: np.ndarray
        2d array with (objects, properties)

    Returns:
    --------
    X_pca: np.ndarray
        Data with PCA components
    """
    # Data matrix X, assumes 0-centered
    X = X - X.mean(axis=0)
    n, m = X.shape
    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)
    return X_pca
