#This method contains methods for statistical analyses

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
