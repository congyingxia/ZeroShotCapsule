import numpy as np
import scipy.spatial.distance as ds

def norm_matrix(matrix):
      """ normalize matrix by column
	input : numpy array, dtype = float32
	output : normalized numpy array, dtype = float32
      """
      # check dtype of the input matrix
      np.testing.assert_equal(type(matrix).__name__, 'ndarray')
      np.testing.assert_equal(matrix.dtype, np.float32)

      row_sums = matrix.sum(axis=1)
      # replace zero denominator
      row_sums[row_sums == 0] = 1
      norm_matrix = matrix / row_sums[:, np.newaxis]
      return norm_matrix

def replace_nan(X):
    """ replace nan and inf to 0
    """
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X

def compute_label_sim(sig_y1, sig_y2, sim_scale):
    """ compute class label similarity
    """
    dist = ds.cdist(sig_y1, sig_y2, 'euclidean')
    dist = dist.astype(np.float32)
    Sim = np.exp(-np.square(dist) * sim_scale);
    s = np.sum(Sim, axis=1)
    Sim = replace_nan(Sim / s[:, None])
    return Sim
