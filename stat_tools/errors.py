#This module contains methods for error and covariance analyses.

import numpy as np

def vec_cov(x, y, normed = False):
    """
    This method computes the covariance between vectors x and y.

    Parameters:
    -----------
    x, y: np.ndarray, np.ndarray
        2d arrays (n,m), with n realizations of a m size vector
    normed: bool
        If True, the normalized covariance is obtained

    Returns:
    --------
    cov_ij: np.ndarray
        Covariance
    """
    cov_ij = np.mean(np.dot((x - np.mean(x)),(y - np.mean(y)).T))
    if normed:
        cov_ii = np.mean(np.dot((x - np.mean(x)),(x - np.mean(x)).T))
        cov_jj = np.mean(np.dot((y - np.mean(y)),(y - np.mean(y)).T))
        return cov_ij/np.sqrt(cov_ii*cov_jj)
    else:
        return cov_ij

def vec_cov_mat(x, y, normed = False):
    """
    This method computes the covariance matrix of vector variables x and y.

    Parameters:
    -----------
    x, y: np.ndarray, np.ndarray
        2d arrays (l,n,m), with l realizations of n variables of m size vector
    normed: bool
        If True, the normalized covariance is obtained

    Returns:
    --------
    cov: np.ndarray
        Covariance matrix
    """
    n_var = x.shape[1]
    cov = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(n_var):
            cov[i,j] = vec_cov(x[:,i], y[:,j], normed = normed)
    return cov

def get_jk_indeces_1d(array, jk_num, rand_order = True):
    """
    This method assigns equally distributed indeces to the elements of an array.

    Parameters:
    -----------
    array: np.array
        Data array
    jk_num: int
        Number of JK subsamples to identify
    rand_order: bool
        If True, the indeces are assigned in a random order

    Returns:
    --------
    jk_indeces: np.array
        Array assigning an index (from 0 to jk_num - 1) to
        each of the data elements
    """
    ratio = int(len(array)/jk_num)
    res = int(len(array)%jk_num > 0)
    jk_indeces = (np.arange(len(array), dtype = int)/ratio).astype(int)
    jk_indeces[-res:] = np.random.randint(jk_num, size = res)#TODO test
    np.random.shuffle(jk_indeces)
    return jk_indeces

def mean_err_jk(array, jk_num = 50, rand_order = True):
    """
    This method measures the mean and Jack-Knife (JK) error of
    the values in an array

    Parameters:
    -----------
    array: np.array
        Data array
    jk_num: int
        Number of JK subsamples to identify
    rand_order: bool
        If True, the indeces are assigned in a random order

    Returns:
    --------
    mean_val: float
        Mean value of the array
    err: float
        JK error of the mean
    """
    mean_val = np.mean(array)
    jk_ids = get_jk_indeces_1d(array, jk_num, rand_order)
    mean_arr_jk = np.array([np.mean(array[jk_ids != i]) for i in range(jk_num)])
    err = jack_knife(np.array([mean_val]), mean_arr_jk)
    return mean_val, err

def jack_knife(var, jk_var):
    """
    This method gives the Jack-Knife error of var from the jk_var subsamples.

    Parameters:
    -----------
    var: float
        The mean value of the variable
    jk_var: np.ndarray
        The variable from the subsamples. The shape of the jk_var must be (jk subsamples, bins)

    Returns:
    --------
    jk_err: float
        The JK error of var.
    """
    jk_dim = np.prod(jk_var.shape)
    err = (jk_dim - 1.)/jk_dim * (jk_var - var)**2.
    jk_err = np.sqrt(np.sum(err, axis = 0))
    return jk_err

def sig2pow(sig):
    """
    This method returns the confidence interval
    of a distribution from a sigma factor.

    Parameters:
    -----------
    sig: float
        Value indicating how many sigmas away the
        signal is.

    Returns:
    --------
    p: float
        Power indicating interval of confidence.
    """
    return sci_esp.erf(sig/np.sqrt(2.))

def bootstrap_resample(data):
    """
    This method creates a shuffled version of the data
    with resampling (so repetitions can happen).

    Parameters:
    -----------
    data: np.ndarray
        Data with shape (samples, values)

    Returns:
    --------
    new_data: np.ndarray
        New data resample from the original, resampling
        the samples with their data
    """
    data_len = len(data)
    rand_ints = np.random.randint(0, data_len, data_len)
    new_data = data[rand_ints]
    return new_data

def boostrap_mean_err(data, nrands = 100):
    """
    This method calculates the mean and error of an array of values
    using the Bootstrap method.

    Parameters:
    -----------
    data: np.ndarray
        A 1-d array of values
    nrands: int
        Number of Bootstrap iteration to calculate the error

    Returns:
    --------
    mean: float
        Mean value of the data
    err: float
        Bootstrap error of the mean
    mean_resamples: float
        Mean over all the resamples
    """
    means = np.zeros(nrands)
    mean = np.mean(data)
    for i in range(nrands):
        r_data = bootstrap_resample(data)
        means[i] = np.mean(r_data)
    err = np.std(means)
    mean_resamples = np.mean(means)
    return mean, err, mean_resamples

def chi_square(var_1, err_1, var_2, err_2, use_err = True):
    """
    This method calculates chi2 value of two between two variables with their errors.

    Parameters:
    -----------
    var_1: np.array
        Values of the first variable
    err_1: np.array
        Errors on the first variable
    var_2: np.array
        Values of the second variable
    err_2: np.array
        Errors on the second variable
    use_err: if False, errors are ignored in the calculation.

    Returns:
    --------
    chi2: float
        Chi square value between the two variables
    """
    if var_1.shape == err_1.shape and var_1.shape == var_2.shape and var_1.shape == err_2.shape:
        mask = np.isfinite(var_1)&np.isfinite(var_2)
        if use_err:
            mask = mask&np.isfinite(err_1)&np.isfinite(err_2)
            chi2 = np.mean((var_1[mask] - var_2[mask])**2. /(err_1[mask]**2. + err_2[mask]**2))
            return chi2
        else:
            chi2 = np.mean((var_1[mask] - var_2[mask])**2.)
            return chi2
    else:
        print("All the variables of chi_square must have the same shape")
