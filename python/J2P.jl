using SparseArrays
using PyCall
using CoDeBetHe



function ComDet(values, idx1, idx2, shape1, shape2; k = nothing, verbose = 1)
    
    A = sparse(idx1 , idx2, values, shape1, shape2)
    res = community_detection_optimal_BH(A, k = k, verbose = verbose) 

    return res.ℓ, res.k, res.overlap, res.modularity, res.ζ

end