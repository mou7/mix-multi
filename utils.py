import numpy as np

def normalize_mat(M):
    """
    This function normalizes the matrix M so the sum of its elements along
    each row is 1

    Arguments:
    - `M`: numpy array
    """
    if len(M.shape) ==2:
      M = 1.0*np.array(M)
      s =  np.sum(M,axis=1)
      index = np.nonzero(s)[0]
      s = s[index]
      s = s.reshape(s.size,1,order='F')
      M[index,:] = M[index,:] / np.tile(s,(1,M.shape[1]))
      return M
    else:
      if len(M.shape)==1:
        return M/np.sum(M)