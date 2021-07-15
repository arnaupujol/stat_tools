#This method contains methods for statistical analyses

import numpy as np
import scipy.stats as sci_stats

def mat_mean(mat):
    """
    This method computes the mean value of a diagonally
    symmetric matrix, excluding the diagonal
    matrix and one of the symmetric halfs.

    Parameters:
    -----------
    mat: np.ndarray
        Matrix with diagonaly symmetry.

    Returns:
    --------
    mean: float
        Mean value of matrix.
    """
    mat_len = len(mat)
    total = .5*(mat_len**2. - mat_len)
    mean = .0
    for i in range(mat_len):
        for j in range(mat_len):
            if j > i:
                mean += mat[i,j]/total
    return mean

def mat_vals(mat, mask = None, diag = True):
    """
    This method returns the values in a
    diagonally symmetric matrix excluding
    the diagonal terms and the elements masked.

    Parameters:
    -----------
    mat: np.ndarray
        Matrix with diagonaly symmetry.
    mask: np.array
        Mask selecting the elements in mat used.
    diag: bool
        It specifies whether the matrix is diagonally
        symmetric and repeated cases are excluded

    Returns:
    --------
    vals: np.array
        array of all the non-diagonal values
        of the matrix.
    """
    vals = []
    mat_len = mat.shape
    if mat_len[0] != mat_len[1]:
        diag = False
    if mask is None:
        if diag:
            mask = np.ones(mat_len[0], dtype = bool)
        else:
            mask = np.ones(mat_len, dtype = bool)
    for ii in range(mat_len[0]):
        for jj in range(mat_len[1]):
            if diag:
                if jj > ii and mask[ii] and mask[jj]:
                    vals.append(mat[ii,jj])
            else:
                if mask[ii,jj]:
                    vals.append(mat[ii,jj])
    return np.array(vals)

def pearson_cc_boostrap(data1, data2, nrands = 100):
    """
    This method calculates the pearson correlation coefficient
    between two variables and its error using the Bootstrap method.

    Parameters:
    -----------
    data1: np.ndarray
        A 1-d array of values
    data1: np.ndarray
        Another 1-d array of values
    nrands: int
        Number of Bootstrap iteration to calculate the error

    Returns:
    --------
    pcorr: float
        Pearson correlation coefficient of the data
    conf_95: float
        The 95% confidence interval of the Pearson correlation
        coefficient
    conf_68: float
        The 68% confidence interval of the Pearson correlation
        coefficient
    pcorr_resamples: float
        Mean Pearson correlation coefficient over all the resamples
    """
    pcorrs = np.zeros(nrands)
    pcorr = sci_stats.pearsonr(data1, data2)[0]
    for i in range(nrands):
        data_len = len(data1)
        rand_ints = np.random.randint(0, data_len, data_len)
        r_data1 = data1[rand_ints]
        r_data2 = data2[rand_ints]
        pcorrs[i] = sci_stats.pearsonr(r_data1, r_data2)[0]
    err = np.std(pcorrs)
    sorted_pcorrs = np.sort(pcorrs)
    conf_95 = [sorted_pcorrs[int(nrands*.05)], sorted_pcorrs[int(nrands*.95)]]
    conf_68 = [sorted_pcorrs[int(nrands*.32)], sorted_pcorrs[int(nrands*.68)]]
    pcorr_resamples = np.mean(pcorrs)
    return pcorr, err, conf_95, conf_68, pcorr_resamples
