import numpy as np

from julia.api import Julia
jl = Julia(runtime="/home/lorenzo/julia-1.8.5/bin/julia", compiled_modules = False)

jl.eval('using Base')
jl.eval('include("J2P.jl")')
ComDet = jl.eval('ComDet')



def CD_BH(A, k = None, verbose = 0):
    '''
    This function performs Community detection using the Bethe-Hessian matrix as per (Dall'Amico et al 2021)

    Use: ℓ, k, modularity, ζ = CD_BH(A)

    Inputs: 
        * A (scipy sparse array): Adjacency matrix of the input graph

    Optional inputs:
        * k (int): number of communities. If not specified (default), it will be estimated
        * verbose (int): sets the level of verbosity of the algorithm

    Outputs:
        *ℓ (array): estimated label partition
        * k (int): number of communties
        * modularity (float): modularity of the partition
        * ζ (array): optimal zeta values used in the algorithm

    '''

    idx1, idx2 = A.nonzero()
    values = np.array(A[idx1, idx2])[0]
    shape1, shape2 = A.shape

    idx1 = (idx1+1).astype(np.int64)
    idx2 = (idx2+1).astype(np.int64)

    mod = ComDet(values, idx1, idx2, shape1, shape2, verbose = verbose, k = k)

    return mod[0], mod[1], mod[3], mod[4]
