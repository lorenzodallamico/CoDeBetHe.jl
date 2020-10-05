module CoDeBetHe

using Distributions
using LinearAlgebra
using DataFrames
using StatsBase
using IterativeSolvers
using Clustering
using SparseArrays
using KrylovKit
using LightGraphs
using DelimitedFiles
using ParallelKMeans


export create_label_vector, matrix_C, adjacency_matrix_DCSBM, community_detection_optimal_BH, adjacency_matrix_DDCSBM, dynamic_community_detection_BH

"""
For a given diagonal matrix Π, with Tr(Π) = 1, this function generates a class affinity matrix C such that CΠ
has the all ones vector as leading eigenvector with eigenvalue equal to c. The parameter f allows to add randomness
in the elements of C ∈ R^{k×k}

Usage
----------
```C = matrix_C(c_out,c,f,π_v)```


Entry
----------
* ```c_out```: average value of the off-diagonal terms  of the matrix ```C``` (Float64)
* ```c``` : leading eigenvalue of ```CΠ``` (Float64)
* ```f``` : variance of the off-diagonal terms of the matrix ```C``` (Float64)
* ```π_v``` : vector of size k containing the diagonal elements of ```Π``` (Array{Float64,1})

Returns
-------
```C``` : matrix C (Array{Float64,2})

"""
function matrix_C(c_out::Float64,c::Float64,f::Float64,π_v::Array{Float64,1})
    k = length(π_v) # number of communities
    Π = Diagonal(π_v) # diagonal matrix Π
    C = abs.(rand(Normal(c_out, c_out*f),(k,k))) # generate off-digonal terms
    C = (C + transpose(C))/2 # symmetrize C
    v = C*Π*ones(k)
    w = zeros(k)
    for i = 1:k
        w[i] += (c-v[i])/π_v[i] # modulate the diagonal values so that CΠ1_k = c1_k
    end
    return C + Diagonal(w)
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


"""
This function generates a label vector of size n, given the class sizes

Usage
----------
``` ℓ = create_label_vector(n, k, π_v)```


Entry
----------
* ```n``` : number of nodes (Int64)
* ```k``` : number of classes (Int64)
* ```π_v``` : vector of size ```k```; the ```i```-th entry corresponds to the fraction of nodes with  label equal to ```i```,
            so that ```∑ π_v = 1``` (Array{Float64,1})

Returns
-------
```ℓ``` : label vector (Array{Int64,1})

"""
function create_label_vector(n, k, π_v)
    ℓ = zeros(n)
    for i = 1:k
        a = Int(floor(n*sum(π_v[1:i-1])))+1
        b = Int(floor(n*sum(π_v[1:i])))
        for j = a:b
            ℓ[j] = i
        end
    end
    ℓ[end] = k
    ℓ = convert(Array{Int64}, ℓ)
    return ℓ
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
This function generates the sparse representation of an adjacency matrix A ∈ R^{n×n} according to the degree-corrected
stochastic block model.

Usage
----------
```A = adjacency_matrix_DCSBM(C::Array{Float64,2},c::Float64,ℓ::Array{Int64,1},θ::Array{Float64,1})```


Entry
----------
* ```C``` : class affinity matrix ```C``` (Array{Float64,2})
* ```c``` : average degree (Float64)
* ```ℓ``` : label vector of size ```n``` (Array{Int64,1})
* ```θ``` : vector generating an arbitrary degree distribution. The value ```cθ_i``` is the expected degree of node ```i``` (Array{Float64,1})

Returns
-------
```A``` : sparse representation of the adjacency matrix (SparseMatrixCSC{Float64,Int64})

"""
function adjacency_matrix_DCSBM(C::Array{Float64,2},c::Float64,ℓ::Array{Int64,1},θ::Array{Float64,1})
    k = length(unique(ℓ)) # number of classes, k
    n = length(θ) # number of nodes, n
    fs = []
    ss = []
    M = zeros(n,k) # matrix containing the affinity between each node and each class
    for i=1:n
        for j=1:k
            M[i,j] = C[ℓ[i],j]
        end
    end

    first = sample(collect(1:n), Weights(θ/n),Int(n*c)) # select nc nodes w.p. θ/n

    for i=1:k
        v = θ.*M[:,i]
        first_selected = first[ℓ[first] .== i] # of the nc selected nodes consider those with ℓ = i
        append!(fs,first_selected)
        second_selected  = sample(collect(1:n), Weights(v./sum(v)),length(first_selected)) # select the nodes to connect to the considered ones
        append!(ss,second_selected)
    end

    idx = fs.> ss # keep only edges (ij) in which i>j
    fs2 = fs[idx]
    ss2 = ss[idx]

    fs3 = vcat(fs2,ss2) # symmetrize the edges (ij), (ji)
    ss3 = vcat(ss2,fs2)

    edge_list = hcat(fs3,ss3) # create edge list
    edge_list = Array(unique(DataFrame(edge_list))) # remove repeated edges
    A = sparse(edge_list[:,1],edge_list[:,2], ones(length(edge_list[:,1])), n,n) # create sparse adjacency matrix

    return A
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

struct output_SC_BH{est_ℓ, k, overlap, mod,ζ}
    ℓ::est_ℓ
    k::k
    overlap::overlap
    modularity::mod
    ζ::ζ
end

"""
This function performs community detection of a graph, according to Algorithm 2 of (Dall'Amico 2020)


Usage
----------
```cluster = community_detection_optimal_BH(A; k, ℓ, ϵ, projection, k_max, verbose)```


Entry
----------
```A``` : sparse representation of the adjacenccy matrix (SparseMatrixCSC{Float64,Int64})

Optional inputs
----------

* ```k``` : number of classes (Int64). If not provided, it is estimated
* ```ℓ``` : ground-truth vector (Array{Int32,1}). If not provided, the overlap will not be computed.
* ```ϵ``` : precision error (Float64). If not provided, it is set to 2*10^(-5)
* ```projection``` : (Bool) if true, the embedding on which k-means is run will be projected on the unitary hypersphere.
                   Default is true.
* ```verbose``` : (0, 1, or 2) if 0, nothing is printed. If 1, some information is printed. If 2, more information is printed. Default is 1.


Returns
-------
* ```cluster.ℓ``` : estimated assignement vector (Array{Int64,1})
* ```cluster.k``` : number of classes obtained (Int64)
* ```cluster.overlap``` : overlap obtained (Float64)
* ```cluster.modularity``` : modularity obtained (Float64)
* ```cluster.ζ``` : vector containing the values of ζ_p for 2≤p≤k (Array{Float64,1})

"""
function community_detection_optimal_BH(A::SparseMatrixCSC{Float64,Int64}; k = nothing, ℓ = nothing, ϵ = 2e-5, projection = true, verbose=1)

    n=size(A,1)
    if verbose >= 1 && ϵ < 2e-5
    printstyled("\nNOTE: The required precision ϵ is below the precision of the LOGPCG routine ϵ = 2*1e-5: the output might be affected by an error larger than the required ϵ\n "; color=3)
    end

    if verbose >= 1;  printstyled("\no Computing the largest eigenvalue of the non-backtracking matrix : "; color=4); end
    ρ = ρ_B(A, ϵ) # compute the spectral radius of the matrix B
    if verbose >= 1;  print("found ρ = ", string(ρ), ".\n"); end

    if k == nothing # if k is not known in advance, it is estimated
        if verbose >= 1;  printstyled("\no Estimating the number of clusters:\n"; color=4); end
        k = estimate_number_of_clusters(ρ, A, verbose)
        if verbose >= 1;  print("Estimated number of clusters: ", string(k), "\n"); end
    end

    if verbose >= 1;  printstyled("\no Estimating the parameters ζ and their corresponding informative eigenvectors:\n"; color=4); end
    ζ, Y = find_zeta(A, ρ, k, ϵ, verbose) # compute the eigenvectors associated to the informative eigenvalues

    if projection == true  # normalize the rows of the matrix Y
        for i=1:n
            Y[i,:] = Y[i,:]./sqrt(sum(Y[i,:].^2))
        end
    end

    if verbose >= 1;  printstyled("\no Running k-means\n"; color=4); end
    N_repeat = 5 #repeating kmeans a few times and pick the best solution to avoid bad loacal minima
    fKM = [ParallelKMeans.kmeans(Y', k) for r in 1:N_repeat]
    f = [fKM[r].totalcost for r=1:N_repeat]
    best = argmin(f)
    KM = fKM[best]
    @assert length(unique(KM.assignments)) == k # making sure it did output k clusters
    estimated_ℓ = KM.assignments # perform k-means on the embedding induced by the rows of Y

    if ℓ != nothing
        overlap = compute_overlap(ℓ, estimated_ℓ) # compute the overlap with respect to the ground-truth
    else
        overlap = "Not available"
    end

    mod = modularity(A, estimated_ℓ) # compute the modularity of the assignement

    if verbose >= 1;  printstyled("\nDone\n\n"; color=4); end

    if (k != nothing) && (sum( abs.(ζ .- sqrt(ρ)) .< ϵ ) >= 1) # if k was given but that at least one ζ parameter is equal to sqrt(ρ)
        if verbose >= 1;  printstyled("Note: You have asked for k = ", string(k), " clusters and the results returned were computed with this choice. However, it seems that this is too many. We suggest to re-run the algorithm with k = ", string(sum( abs.(ζ .- sqrt(ρ)) .> ϵ ) + 1 ), " and compare obtained results.\n\n"; color=3); end
    end

    return output_SC_BH(estimated_ℓ, k, overlap, mod, ζ)
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


"""
This function computes the largest (in magnitude) eigenvalue of the non-backtracking matrix of the graph
associated to the adjacency matrix A. It is necessarily real and larger than 1.

Usage
----------
```ρ = ρ_B(A)```


Entry
----------
```A```: the adjacency matrix (SparseMatrixCSC{Float64,Int64})

Returns
-------
```ρ``` : value of ```ρ``` (Float64)

"""
function ρ_B(A::SparseMatrixCSC{Float64,Int64}, ϵ::Float64)
    n = size(A)[1]
    D = spdiagm(0 => A*ones(n)) # diagonal matrix of degrees
    Id = spdiagm(0 => ones(n)) # the identity matrix
    M = [A Id - D; Id spzeros(n, n)] # create M: its largest eigenvalue is known to be the largest eigenvalue of the nonbacktracking (that we don't explicitly build)
    # compute its largest eigenvalue in magnitude:
    ρ, _ = KrylovKit.eigsolve(M, 1, :LM, tol=ϵ)
    return Real(ρ[1])
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################



"""
This function estimates the number of communities by counting the eigenvalues of L_{ρ-1} that are larger than 1/√ρ.

Usage
----------
```estimate_number_of_clusters(k_max, ρ, A, verbose)```

Entry
----------
* ```k_max```: maximal number of classes admitted (Int64)
* ```ρ``` : leading eigenvalue of the matrix B (Float64)
* ```A``` : sparse representation of the adjacency matrix (SparseMatrixCSC{Float64,Int64})
* ```verbose``` : (0, 1, or 2) if 0, nothing is printed. If 1, some information is printed. If 2, more information is printed.

Returns
-------
```k``` : number of estimated classes (Int64)

"""
function estimate_number_of_clusters(ρ::Float64, A::SparseMatrixCSC{Float64,Int64}, verbose::Int64)
    n = size(A)[1] # size of the network
    d = A*ones(n) # degree vector
    D_05 = spdiagm(0 => (d .+ (ρ-1)).^(-1/2)) # regularized degree matrix
    L_ρ = D_05*A*D_05 # regularized Laplacian

    Id = spdiagm(0 => ones(n)) # identity matrix

    if verbose >= 1;  print("Estimating k_neg, the number of negative eigenvalues of M = -L_ρ + 1/sqrt(c) * Id: \n"); end
    λ_neg, _ = estimate_neg_eigs(-L_ρ + 1/sqrt(ρ) * Id, verbose)

    return length(λ_neg)
end


"""
This function computes the eigenspace associated to negative eigenvalues of a sparse symmetric matrix M

Usage
----------
```estimate_neg_eigs(M, verbose)```

Entry
----------
* ```M``` : sparse symmetric matrix (SparseMatrixCSC{Float64,Int64})
* ```verbose``` : (0, 1, or 2) if 0, nothing is printed. If 1, some information is printed. If 2, more information is printed.

Returns
-------
* ```λ_neg``` : vector of negative eigenvalues of M
* ```X_neg``` : associated eigenvectors

"""
function estimate_neg_eigs(M::SparseMatrixCSC{Float64,Int64}, verbose)

    n = size(M,1)
    ϵ = 2e-5
    if verbose >= 1;  print("Computing a first estimate for k_neg: "); end
    sq_sum, order_poly = eigencount_via_poly_approx(M, 0.) # rough estimate of the number of negative eigenvalues
    if order_poly == nothing
        if sq_sum == 0
            print("there is no negative eigenvalue!")
            return 0, nothing
        else sq_sum == n
            print("all eigenvalues are negative! There is very probably a problem, we can't diagonalize the whole matrix!")
            return n, nothing
        end
    end
    k_est = Int(ceil(sum(sq_sum .* (order_poly ./ sum(order_poly))))) # a weighted mean (the larger order_poly, the more confident we can be wrt to the estimation)
    if verbose >= 1;  print("found k_neg = ", string(k_est), " (with a polynomial approximation with a mean polynomial order of ", string(Int(ceil(mean(order_poly)))), ").\n"); end

    k = k_est # initial guess for the number of negative eigenvalues

    if verbose >= 1;  print("Trying for k_neg = ", string(k), "\n"); end
    num_iter = 10 # we start with only a few number of iterations as this first try for k_neg might lead us into the bulk of uninformative eigenvalues (thus requiring unnecessary many Krylov iterations to converge)
    λ, X, info = KrylovKit.eigsolve(M, k, :SR, maxiter = num_iter, krylovdim = max(KrylovDefaults.krylovdim, 2*k), tol=ϵ)
    converged = info.normres .< ϵ # Boolean vector with true in position i if eigenvalue λ[i] has converged in num_iter iterations
    rk_largest_neg_eig = findlast(λ .< 0) # rank of the largest eigenvalue in λ that is negative

    if (rk_largest_neg_eig+1 <= length(λ)) && all(converged[1:rk_largest_neg_eig+1]) # if all negative eigenvalues have converged, as well as the first positive eigenvalue
        return λ[1:rk_largest_neg_eig], X[1:rk_largest_neg_eig]
    else
        if !all(converged[1:rk_largest_neg_eig]) # if all negative eigenvalues detected did not converge, unconstrain maxiter and find them:
            k = rk_largest_neg_eig
            if verbose >= 1;  print("Trying for k_neg = ", string(k), "\n"); end
            λ, X, info = KrylovKit.eigsolve(M, k, :SR, maxiter = KrylovDefaults.maxiter, krylovdim = max(KrylovDefaults.krylovdim, 2*k), tol=ϵ)
            converged = info.normres .< ϵ # Boolean vector with true in position i if eigenvalue λ[i] has converged
            rk_largest_neg_eig = findlast(λ .< 0)
            if (rk_largest_neg_eig+1 <= length(λ)) && all(converged[1:rk_largest_neg_eig+1]) # if all negative eigenvalues have converged, as well as the first positive eigenvalue
                return λ[1:rk_largest_neg_eig], X[1:rk_largest_neg_eig]
            end
        end
        num_iter = info.numiter
        # at this point, we know that k eigenvalues have converged in num_iter iterations and that they are all negative
        # we now increment k until the largest eigenvalue crosses 0:
        k = 1 + rk_largest_neg_eig
        while k <= n
            if verbose >= 1;  print("Trying for k_neg = ", string(k), "\n"); end
            λ, X, info = KrylovKit.eigsolve(M, k, :SR, maxiter = min(num_iter*2, KrylovDefaults.maxiter), krylovdim = max(KrylovDefaults.krylovdim, 2*k), tol=ϵ)
            converged = info.normres .< ϵ
            λ = λ[converged]
            X = X[converged]
            if length(λ) < k #this typically happens in large matrices where the k-th smallest eigenvalue is uninformative and thus not isolated (in the bulk)
                if verbose >= 2;  print("The ", string(k), "-th smallest eigenvalue did not converge in ", string(min(num_iter*2, KrylovDefaults.maxiter)), " iterations (while the ", string(k-1), " first did converge in ", string(num_iter), " iterations): decision is made that it is in the bulk of uninformative eigenvalues and that k_neg is thus ", string(k-1), ".\n"); end
                #@assert length(λ) == k-1
                return λ, X
            else
                rk_largest_neg_eig = findlast(λ .< 0)
                if maximum(λ) > 0 #if the largest uncovered eigenvalue is positive
                    return λ[1:rk_largest_neg_eig], X[1:rk_largest_neg_eig]
                else #if the largest uncovered eigenvalue is still negative, then increment k
                    num_iter = info.numiter
                    k = length(λ) + 1
                end
            end
        end
    end
end

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


"""
This function estimates, via polynomial approximation, the number of eigenvalues of H that are smaller than c

Usage
----------
```est, order_poly = eigencount_via_poly_approx(H, c)```

Entry
----------
* ```H``` : square sparse matrix (SparseMatrixCSC{Float64,Int64})
* ```c``` : scalar (Float64)

Optional inputs
----------

* ```upper_bound_method``` : ("Krylov" or "Gershgorin"). It describes the method used to obtain the largest eigenvalue of H. "Krylov" computes it directly, while
                             Gershgorin gives an estimate which is faster but less accurate. By default set to "Krylov".

Returns
-------
* ```est``` : a vector of independent estimates
* ```order_poly``` : a vector of the polynomial order used for each estimate
"""
function eigencount_via_poly_approx(H::SparseMatrixCSC{Float64,Int64}, c::Float64; upper_bound_method="Krylov")
    m = 1000 # the maximal order of the polynomial that will be used

    n = size(H)[1] # size of matrix H
    Id = spdiagm(0 => ones(n)) # identity matrix

    Nv = 3 # number of independent estimations

    ## shift H to make it SDP:
    λmin, _ = KrylovKit.eigsolve(H, 1, :SR, tol=2e-5)
    λmin = minimum(Real.(λmin))
    if λmin > c - 1e-14
        print("there is no eigenvalue smaller than ", string(c), " in H! The smallest eigenvalue found is ", string(λmin), "\n")
        return 0, nothing
    end
    H -= λmin * Id
    # H is now SDP: its first eig is null and all its eigs are >=0.
    # Also, the number of eig smaller than c of the original H is equal to the numbe of eig smaller than c - λmin in the shifted H

    if upper_bound_method == "Krylov"
        λmax, _, _ = KrylovKit.eigsolve(H, 1, :LR, krylovdim=20, tol=2e-3)
        λmax = (1+2e-3) * maximum(λmax) #to have an extra assurance we have an upper bound
    elseif upper_bound_method == "Gershgorin" # use Gershgorin's upper bound
        λmax = maximum(vec(sum(abs.(H - spdiagm(0=>diag(H))), dims=1)) + diag(H))
    end

    if c - λmin > λmax
        print("All eigenvalues are below the threshold!\n")
        return n, nothing
    end

    # compute the polynomial coefficients of the jackson-Chebychev approximation to the ideal low-pass with cut-off c - λmin:
    ch, jch = jackson_cheby_poly_coefficients(c - λmin, λmax, m)

    random_matrix = randn(n, Nv) # create Nv random Gaussian vectors of size n
    return est_k_cheby(H, ch, random_matrix, λmax) # apply the CH (or JCH by changing ch in jch) poly of H to each of the columns of random_matrix and output their norm
end

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
This function computes the coefficients of the chebychev polynomial of order m that best approximates
the ideal low pass function with threshold c (i.e., the function f(x) defined on [0, λmax]
that returns 1 if x < c and 0 if not).
It also computes its Jackson-chebychev counterpart (damped polynomial to avoid the
Gibbs oscillations of the Chebychev polynomial)
This function is adapted from the function jackson_cheby_poly_coefficients of the Compressive Spectral Clustering Toolbox

Usage
----------
```ch, jch = jackson_cheby_poly_coefficients(b, λmax, m)```

Entry
----------
* ```b``` : the cut-off of the filter. Must verify ```0 < b < λmax``` (Float64)
* ```λmax``` : the range up to which one whishes the approximation. Must be positive (Float64)
* ```m``` : the order of the polynomial (Int64)

Returns
-------
```ch, jch``` : returns two vectors of m+1 polynomial coefficients. ```ch``` for cheby, ```jch``` for jackson-cheby.
"""
function jackson_cheby_poly_coefficients(b::Float64, λmax::Float64, m::Int64)
    @assert λmax > 0
    @assert 0 < b < λmax
    a = 0
    lambda_range = [0, λmax]

    a1 = (lambda_range[2]-lambda_range[1])/2
    a2 = (lambda_range[1]+lambda_range[2])/2

    a=(a-a2)/a1
    b=(b-a2)/a1

    CH = zeros(m+1)
    CH[1]=(1/pi)*(acos(a)-acos(b))
    for j=2:m+1
        CH[j]=(2/(pi*(j-1)))*(sin((j-1)*acos(a))-sin((j-1)*acos(b)))
    end

    # we have now computed the Chebychev coefficients. For the Jackson Chebychev ones, we need to compute the damping coefficients:
    gamma_JACK = zeros(m+1)
    alpha=pi/(m+2)
    for j=1:m+1
        gamma_JACK[j]=(1/sin(alpha))*((1-(j-1)/(m+2))*sin(alpha)*cos((j-1)*alpha)+(1/(m+2))*cos(alpha)*sin((j-1)*alpha))
    end

    JCH = CH .* gamma_JACK

    # just to be in accordance with the function est_k_cheby (a matter of convention):
    JCH[1] = JCH[1] * 2
    CH[1] = CH[1] * 2

    return CH, JCH
end

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
For each column z in the entry-matrix ```random_matrix```, this function first computes the
chebychev polynomial of H (with coefficients ch) applied to z, and outputs its norm.
The polynomial order finally used is in general lower than the size of the entry ch as
there is a stopping criterion when the result converges.
This function is adapted from the function gsp_cheby_op of the Graph Signal Processing Toolbox.

Usage
----------
```res = est_k_cheby(H, ch, random_matrix, λmax)```

Entry
----------
* ```H``` : square sparse matrix (SparseMatrixCSC{Float64,Int64})
* ```ch``` : vector of polynomial coefficients (Array{Float64,1}). Must be at least of size 80.
* ```random_matrix``` : matrix of random vectors (Array{Float64,2}). Its number of lines must be the number of lines of H
* ```λmax``` : the range up to which one whishes the approximation. Must be positive (Float64)

Returns
-------
```res``` : a vector of size the number of columns of the entry ```random_matrix```.
"""
function est_k_cheby(H::SparseMatrixCSC{Float64,Int64}, ch::Array{Float64,1}, random_matrix::Array{Float64,2}, λmax::Float64; return_vecs = false)
    N = size(H,1)
    M = size(ch,1)

    @assert size(random_matrix,1) == N
    @assert λmax > 0
    @assert M > 80 #needs at least an order of 80 to reasonably approximate the ideal low-pass

    λmin = 0.

    Nv = size(random_matrix,2)

    arange = [λmin, λmax]

    a1 = (arange[2] - arange[1]) / 2
    a2 = (arange[2] + arange[1]) / 2

    sq_sum = zeros(Nv)
    order_poly = zeros(Nv)
    if return_vecs
        vecs = zeros(N, Nv)
    end

    factor = (2/a1) * (H - a2 * spdiagm(0 => ones(N)))

    for i=1:Nv
        twf_old = random_matrix[:,i]
        twf_cur = (H * twf_old - a2 * twf_old) / a1
        r = (0.5 * ch[1]) * twf_old + ch[2] * twf_cur

        for k = 2:79 # we do the first 80 rounds in any case (we suppose that we need at least that)
            twf_new = factor * twf_cur - twf_old
            r += ch[k+1] * twf_new
            twf_old = twf_cur
            twf_cur = twf_new
        end

        # for the next ones, we do not go necessarily to M-1 and stop if the result converges before
        srold = norm(r)^2
        k = 79
        δ = 1
        while (δ > 0.01) && (k < M-1)
            k += 1
            twf_new = factor * twf_cur - twf_old
            r += ch[k+1] * twf_new
            twf_old = twf_cur
            twf_cur = twf_new
            srnew = norm(r)^2
            δ = abs(srnew - srold)# / srnew
            srold = srnew
            #print("for i = ", string(i), " : ", string(srold) ,"\n")
        end
        #print("Vector number i = ", string(i), " : converged for m = ", string(k), ".\n")
        sq_sum[i] = srold
        order_poly[i] = k
        if return_vecs
            vecs[:,i] = r
        end
    end
    if return_vecs
        return vecs, sq_sum, order_poly
    else
        return sq_sum, order_poly
    end
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


"""
This function finds the solution to the equation of line 8 in Subroutine 2 of (Dall'Amico 2020)

Usage
----------
```rp = find_solution(S,M,r,ϵ)```


Entry
----------
* ```S``` : diagonal matrix containing the first ```p``` eigenvalues of ```H_r``` (Diagonal{Float64,Array{Float64,1}})
* ```Λ``` : matrix defined in line 7 of Subroutine 2 (Array{Float64,2})
* ```r```: value of r (Float64)
* ```ϵ``` : precision error (Float64)

Returns
-------
```rp``` : solution to the fixed point (Float64)

"""
function find_solution(S::Diagonal{Float64,Array{Float64,1}},Λ::Array{Float64,2},r::Float64,ϵ::Float64)
    r_small = 1 # left edge
    r_large = r # right edge
    err = 1 # err
    r_old = r_large

    while err > ϵ
        r_new = (r_small+r_large)/2 # mid point
        err = abs(r_old-r_new) # update error
        v = maximum(eigvals(r_new*S + (r-r_new)*Λ))
        if v > (r-r_new)*(1+r*r_new)
            r_small = r_new
        else
            r_large = r_new
        end
        r_old = r_new
    end
    return r_large
end



#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
This function estimates the values of ζ_p for 2 ≤ p ≤ k, defined so that the  p-th smallest eigenvalue of H_{ζ_p}
is equal to zero. The function also computes the corresponding eigenvectors.

Usage
----------
```ζ, Y = find_zeta(A, ρ, k, ϵ, verbose)```


Entry
----------
* ```A``` : sparse representation of the adjacenccy matrix (SparseMatrixCSC{Float64,Int64})
* ```ρ``` : leading eigenvalue of the matrix B (Float64)
* ```k```: number of classes (Int64)
* ```ϵ``` : precision error (Float64)
* ```verbose``` : (0, 1, or 2) if 0, nothing is printed. If 1, some information is printed. If 2, more information is printed.


Returns
-------
* ```ζ``` : vector of size k-1 with the values of ```ζ_p``` for ```2≤p≤k``` (Array{Float64,1})
* ```Y``` : matrix ∈ R^{m×(k-1)} with the informative eigenvectors stored in the columns (Array{Float64,2})

"""
function find_zeta(A::SparseMatrixCSC{Float64,Int64}, ρ::Float64, k::Int64, ϵ::Float64, verbose::Int64)

    Krylovtol = ϵ # set the tolerance of the Krylov routine to ϵ
    n = size(A)[1] # size of the network, n
    D = spdiagm(0 => A*ones(n)) # degree matrix
    Id = spdiagm(0 => ones(n)) # identity matrix

    ζ = zeros(k-1) # initialisation of the ζ vector
    Y = zeros(n,k-1) # initialisation of the node embedding
    r = sqrt(ρ) # initial value of r
    δ_th = max(((r-1) / 1e8) * n, 1.1*ϵ) # arbitrary choice
    if verbose>=2; print("δ_th = ", string(δ_th), ".\n"); end

    i = k # first compute ζ_k

    global X, λ, δ
    while i > 1
        if verbose>=1; print("Estimating ζ_",  string(i), ": "); end
        if i == k #only for i=k, use Krylov for the first attempt
            H = (r^2-1)*Id+D-r*A # Bethe-Hessian associated to parameter r
            λ, X, info = KrylovKit.eigsolve(H, i, :SR, krylovdim = max(KrylovDefaults.krylovdim, 2*i), tol=Krylovtol)
            if (info.converged < i) && (verbose >= 2) # if info.converged < i, not all eigenvalues have converged
                printstyled("Only ", string(info.converged), " eigenvalues out of ", string(i), " have converged. This is probably not problematic. "; color=3)
            end
            S = Diagonal(λ[1:i])
            X = reduce(hcat, X[1:i])
            M = X'*D*X
            rp = find_solution(S, M, r, ϵ)
            δ = abs(r-rp)
            if verbose >= 2; print("δ = ", string(δ), " : "); end
            r = rp
        else
            δ = 0.99 * δ_th  # this is just to make sure that for i != k, the first attempt will be lobpcg
            if verbose>=2; print("initializing δ to ", string(δ_th), " : "); end
        end

        while δ > ϵ
            H = (r^2-1)*Id+D-r*A # Bethe-Hessian associated to parameter r
            if δ > δ_th #if the change in r is "large"
                if verbose>=2; print("using Krylov \n"); end
                λ, X, info = KrylovKit.eigsolve(H, i, :SR, krylovdim = max(KrylovDefaults.krylovdim, 2*i), tol=Krylovtol)
                if (info.converged < i) && (verbose>=2) # if info.converged < i, not all eigenvalues have converged
                    printstyled("Only ", string(info.converged), " eigenvalues out of ", string(i), " have converged. This is probably not problematic. "; color=3)
                end
                λ = λ[1:i]
                X = reduce(hcat, X[1:i])
            else #if the change in r is "small" enough, it is actually worthwhile to use lobpcg with the previous eigenvectors as initialization
                if verbose>=2; print("using lobpcg \n"); end
                try #in some instances, lobpcg is unstable and the Cholesky factorization fails for some reason
                    eigenv = lobpcg(H, false, X[:, 1:i])
                    if (!eigenv.converged) && (verbose>=2) # if !eigenv.converged == true, not all eigenvalues have converged
                        printstyled("lobpcg did not converge. This is probably not problematic. "; color=3)
                    end
                    λ = eigenv.λ
                    X = eigenv.X
                catch #if lobpcg fails, use Krylov
                    if verbose>=2; printstyled("lobpcg failed. Using Krylov \n"; color=3); end
                    λ, X, info = KrylovKit.eigsolve(H, i, :SR, krylovdim = max(KrylovDefaults.krylovdim, 2*i), tol=Krylovtol)
                    if (info.converged < i) && (verbose>=2) # if info.converged < i, not all eigenvalues have converged
                        printstyled("Only ", string(info.converged), " eigenvalues out of ", string(i), " have converged. This is probably not problematic."; color=3)
                    end
                    λ = λ[1:i]
                    X = reduce(hcat, X[1:i])
                end
            end
            S = Diagonal(λ)
            M = X' * D * X
            rp = find_solution(S, M, r, ϵ)
            δ = abs(r-rp)
            if verbose>=2; print("δ = ", string(δ), " : "); end
            r = rp
        end

        degeneracy = sum(abs.(λ .- maximum(λ)) .< ϵ)
        ζ[i-degeneracy:i-1] .= r # store the last value of r* found
        Y[:,i-degeneracy:i-1] += X[:,i-degeneracy+1:i] # store the corresponding eigenvectors
        if verbose>=1; print("found ζ_",  string(i), " = ", string(round(r,digits = 3)), ". The residual of the associated eigenvector is ", string(round(sqrt.(sum((H * X[:,i-degeneracy+1:i]).^2,dims=1))[1], sigdigits = 2)), "\n\n"); end

        i = i-degeneracy
    end
    return ζ, Y
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
This function computes the overlap between two given label assignments, finding the optimal label permutation.

Usage
----------
```overlap = compute_overlap(ℓ,estimated_labels)```


Entry
----------
* ```ℓ``` : ground-truth label assignement (Array{Int64,1})
* ```estimated_labels``` : estiamated label assignement (Array{Int64,1})


Returns
-------
```overlap``` : value of the overlap (Float64)

"""
function compute_overlap(ℓ::Array{Int64,1},estimated_labels::Array{Int64,1})
    n = length(ℓ)
    k = maximum(ℓ) # number of classes
    k_est = maximum(estimated_labels) # estimated number of classes
    confusion = zeros(k,k_est) # confusion matrix. The entry a,b tells how many nodes are labelled a in ℓ and b in estimated_labels
    for i=1:n
        confusion[ℓ[i],estimated_labels[i]] += 1
    end

    est_ℓ = zeros(n)
    for i=1:k
        wh = findmax(confusion[i,:])[2] # find the label assignement that fits best
        est_ℓ[estimated_labels .== wh] .= i
        confusion[i,:] .= 0
        confusion[:,wh] .= 0
    end
    overlap = (sum(est_ℓ .== ℓ)/n-1/k)/(1-1/k) # compute the overlap

    return overlap
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
This function computes the modularity for a graph and a given label assignement.

Usage
----------
```modularity = modularity(A, est_ℓ)```


Entry
----------
* ```A``` : sparse representation of the adjacenccy matrix (SparseMatrixCSC{Float64,Int64})
* ```est_ℓ``` : estimated label assignement (Array{Int64,1})


Returns
-------
```modularity``` : value of the ovemodularityrlap (Float64)

"""
function modularity(A::SparseMatrixCSC{Float64,Int64}, est_ℓ)
    n = size(A)[1] # size of the network, n
    d = A*ones(n) # degree vector
    m = sum(d)
    k = length(unique(est_ℓ))
    mod = 0
    for i=1:k
        I_i = (est_ℓ .== i)*1.
        mod += I_i'*A*I_i - (d'*I_i)^2/m
    end
    return mod/m
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
# codes specific to the dynamic case
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
This function creates the evolution of the community labels with persistence η according to a Markov process.

Usage
----------
```ℓ_T = dynamic_label_HMM(η, T, k, n, π_v)```


Entry
----------
* ```η``` : label persistence (Float64)
* ```T```: number of time steps (Int64)
* ```k``` : number of communities (Int64)
* ```n``` : number of nodes (Int64)
* ```π_v``` : vector of size k; the i-th entry corresponds to the fraction of nodes with  label equal to ```i```,
            so that ```∑ π_v = 1``` (Array{Float64,1})

Returns
-------
```ℓ_T``` : the entry ```ℓ_T[t,i]``` contains the label of node i at time t (T×n Array{Int64,2})

"""
function dynamic_label_HMM(η::Float64, T::Int64, k::Int64, n::Int64, π_v::Array{Float64,1})

    ℓ_T = zeros(T,n)
    ℓ_T = convert(Array{Int64}, ℓ_T)
    ℓ = create_label_vector(n, k, π_v) # assign the labels at the first time step
    ℓ_T[1,:] +=  ℓ
    for i = 2:T
        select = rand(Binomial(1,1-η), n) # select the nodes whose label will be reassigned
        draw = sample(collect(1:k), Weights(π_v),sum(select .== 1)) # assign the new labels
        ℓ[select .== 1] = draw
        ℓ_T[i,:] += ℓ
    end

    return ℓ_T
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


"""
This function generates a series of T adjacency matrices A_t ∈ R^{n×n} in sparse represention from the dynamical
degree-corrected stochastic block model

Usage
----------
```AT, ℓ_T = adjacency_matrix_DDCSBM(T, C, c, η, θ, π_v)```


Entry
----------
* ```T```: number of time steps (Int64)
* ```C``` : matrix ```C``` (Array{Float64,2})
* ```c``` : average degree (Float64)
* ```η``` : label persistence (Float64)
* ```θ``` : vector generating an arbitrary degree distribution. The value ```cθ_i``` is the expected degree of node ```i```
            (Array{Float64,1})
* ```π_v``` : vector of size k; the i-th entry corresponds to the fraction of nodes with  label equal to ```i```,
            so that ```∑ π_v = 1``` (Array{Float64,1})

Returns
-------
* ```AT``` : ```At[t]``` is the sparse representation of the adjacency matrix at time t (Array{SparseMatrixCSC{Float64,Int64},1})
* ```ℓ_T``` : the entry ```ℓ_T[t,i]``` contains the label of node i at time t (T×n Array{Int64,2})

"""
function adjacency_matrix_DDCSBM(T::Int64, C::Array{Float64,2}, c::Float64, η::Float64, θ::Array{Float64,1}, π_v::Array{Float64,1})

    n = length(θ) # number of nodes
    k = length(π_v) # number of communities
    ℓ_T = dynamic_label_HMM(η, T, k, n, π_v) # generate the label assignement
    AT = [adjacency_matrix_DCSBM(C,c,ℓ_T[i,:],θ) for i=1:T] # generate each adjacency matrix independently according to a DC-SBM

    return AT, ℓ_T
end

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

struct output_dyn_SC_BH{est_ℓ, overlap, mod}
    ℓ::est_ℓ
    overlap::overlap
    modularity::mod
end


"""
This function implements algorithm 1 of (Dall'Amico 2020) for dynamical community detection.

Usage
----------
```cluster = dynamic_community_detection_BH(AT, η, k; ℓ_T, approx_embedding, verbose, m)```


Entry
----------
* ```AT``` : ```At[t]``` is the sparse representation of the adjacency matrix at time ```t```
    (Array{SparseMatrixCSC{Float64,Int64},1})
* ```η``` : label persistence (Float64)
* ```k``` : number of communities (Int64)

Optional inputs
----------

* ```ℓ_T``` : the entry ```ℓ_T[t,i]``` contains the label of node ```i``` at time ```t``` (T×n Array{Int64,2}). If not available, the overlap with the ground-truth is not computed.
* ```approx_embedding``` : (Bool), if true the informative eigenvectors are obtained using the approximate procedure
                         detailed in Algorithm 2 of (Dall'Amico 2020). By default  set to false
* ```verbose``` : (0, 1, or 2) if 0, nothing is printed. If 1, some information is printed. If 2, more information is printed. Default is 1.
*  ```m``` : order  of the polynomial approximation (Int64). If ```m``` is not known, an adaptive choice for ```m``` will be  adopted. By default set to ```nothing```

Returns
----------

* ```cluster.modularity``` : ```cluster.modularity[t]``` is the modularity obtained at time ```t``` (Array{Float64,1})
* ```cluster.overlap``` : ```cluster.overlap[t]``` is the overlap obtained at time ```t``` (Array{Float64,1})
* ```cluster.ℓ``` : ```cluster.ℓ[t,i]``` is the vector of the estimated label i at time ```t``` (Array{Array{Int64,1},1})

-------
"""
function dynamic_community_detection_BH(AT::Array{SparseMatrixCSC{Float64,Int64},1}, η::Float64, k::Int64; ℓ_T = nothing, approx_embedding = false, m = nothing, verbose=1)

    T = length(AT) # number of time steps
    n = length(AT[1][1,:]) # number of nodes
    if verbose >= 1;  printstyled("\no Creating the dynamical Bethe Hessian matrix H (of size nT = ", string(Int64(n*T)), ").\n"; color=4); end
    α_c = find_transition(T,η) # value of α at the detectability threshold
    c = sum([sum(AT[i])/n for i=1:T])/T # average degree
    Φ = sum([sum((AT[i]*ones(n)).^2)/n for i=1:T])/(T*c^2) # estimated value of Φ
    λ_d = α_c/sqrt(c*Φ)
    H = dyn_BH_matrix(AT, λ_d, η)  # dynamical BH
    if approx_embedding
        if verbose >= 1;  printstyled("\no Approximating the eigenspace associated to negative eigenvalues of H:\n"; color=4); end
        X = approximate_embedding(H, verbose, m = m) # approximate the eigenspace associated to neg. eigs. of H
        if verbose >= 1;  print("The computed approximate embedding is of dimension ", string(size(X,2)), ".\n"); end
    else
        if verbose >= 1;  printstyled("\no Computing the eigenvectors associated to all negative eigenvalues of H:\n"; color=4); end
        _, X = estimate_neg_eigs(H, verbose) # compute the negative eigenvectors of H
        if verbose >= 1;  print("The computed embedding is of dimension ", string(length(X)), ".\n"); end
        X = X[2 : end] # exclude the smallest (uninformative) eigenvector
        X = reduce(hcat, X)
    end

    for i=1:n*T
        X[i,:] = X[i,:]/sqrt(sum(X[i,:].^2)) # normalize the rows of X
    end

    if verbose >= 1;  printstyled("\no Computing k-means for each time step\n"; color=4); end
    KM = [ParallelKMeans.kmeans([0.]', 1) for i=1:T] #initialize the vector KM to an array of type::ParallelKMeans.KmeansResult
    # apply k-means independently at each time-step:
    for i=1:T
        N_repeat = 5 #repeating kmeans a few times and pick the best solution to avoid bad local minima
        fKM = [ParallelKMeans.kmeans(X[(i-1)*n+1:i*n,:]', k) for r in 1:N_repeat]
        f = [fKM[r].totalcost for r=1:N_repeat]
        best = argmin(f)
        KM[i] = fKM[best] # keep the best solution
    end

    estimated_ℓ = [KM[i].assignments for i=1:T]
    if ℓ_T != nothing
        overlap = [compute_overlap(ℓ_T[i,:],estimated_ℓ[i,:][1]) for i=1:T] # compute the overlap at each time-step
    else
        overlap = "not available"
    end
    mod = [modularity(AT[i], estimated_ℓ[i]) for i=1:T] # compute the modularity at each time step

    estimated_ℓ = match_labels(estimated_ℓ, verbose)

    return output_dyn_SC_BH(estimated_ℓ, overlap, mod)
end

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
This function matches the label assignement accross time for fixed n and η

Usage
----------
```new_ℓ_T = match_labels(ℓ_T, verbose)```


Entry
----------
*```ℓ``` : the entry ```ℓ[t]``` contains the estimated labels at time t (Array{Array{Int64,1},1})
* ```verbose``` : (0, 1, or 2) if 0, nothing is printed. If 1, some information is printed. If 2, more information is printed. Default is 1.

Returns
-------
```new_ℓ_T``` : the entry ```new_ℓ_T[t,i]``` contains the reassigned label of node i at time t (T×n Array{Int64,2})

"""
function match_labels(ℓ::Array{Array{Int64,1},1}, verbose::Int64)


    T = length(ℓ)
    n = length(ℓ[1])

    k_v = [length(unique(ℓ[t])) for t=1:T] # number of clusters at each time step
    mean_k = mean(k_v)
    x = sum(k_v .== ones(length(k_v))*mean_k)
    if x != length(k_v)
        printstyled("\n Note: the number of classes is not constant. Probably there is a problem\n\n"; color=3)
         e_ℓ = zeros(T,n)
        for t=1:T
            e_ℓ[t,:] = new_labels[t]
        end

        return convert(Array{Int64,2}, e_ℓ)
    end

    new_labels = [zeros(n) for i=1:T]
    new_labels[1] = ℓ[1]
    k = Int(ceil(mean_k))

    for t=1:T-1

        a = convert(Array{Int64},new_labels[t])
        b = ℓ[t+1]
        confusion = zeros(k,k) # confusion matrix. The entry a,b tells how many nodes are labelled a in ℓ and b in estimated_labels
        for i=1:n
            confusion[a[i],b[i]] += 1
        end


        for i=1:k
            wh = findmax(confusion[i,:])[2] # find the label assignement that fits best
            new_labels[t+1][b .== wh] = i*ones(sum(b .== wh)) # assign the new value
            confusion[i,:] .= 0 #zero out the corresponding row and column
            confusion[:,wh] .= 0

        end

        new_labels[t+1][new_labels[t+1] .== 0] = ℓ[t+1][new_labels[t+1] .== 0] # keep label asisgnement for nodes that were not assigned (unlikely to happen)

    end

    e_ℓ = zeros(T,n)
    for t=1:T
        e_ℓ[t,:] = new_labels[t]
    end

    return convert(Array{Int64,2}, e_ℓ)

end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


"""
This function computes the critical value of α at the detectability threshold for η, T given

Usage
----------
```α_c = find_transition(T, η)```


Entry
----------
* ```T``` : number of time steps (Int64)
* ```η``` : label persistence (Float64)

Returns

* ```α_c``` : critical value of ```α``` at the transition (Float64)
-------
"""
function find_transition(T::Int64, η::Float64)

    α_min = 0 # minimal value that α_c can take
    α_max = 1 # maximal value that α_c can take
    err = 1 # initialization of the precision error

    while err > eps()
        global α = (α_min + α_max)/2 # the new α is the mid point between the extrema
        M = M_T(T, α, η) # compute the matrix M_T of equation 4
        λ, X, info = KrylovKit.eigsolve(M, 1, :LR)
        v = maximum(real(λ))
        if v > 1 # update the boundaries
            α_max = α
        else
            α_min = α
        end
        err = (α_max - α_min)/2 # update the error
    end
    return α
end



#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
This function defines the matrix M_T of Equation 4 in (Dall'Amico 2020), for given values of T, η, α.

Usage
----------
```M =  M_T(T, α, η)```


Entry
----------
* ```T``` : number of time steps (Int64)
* ```α``` : value of ```α``` (Float64)
* ```η``` : label persistence (Float64)

Returns
-------

```M``` : Matrix ```M_T``` (Array{Float64,2})

"""
function M_T(T::Int64, α::Float64, η::Float64)

    M_plus = [0 0 0; 0 0 0; 0 α^2 η^2]
    M_minus = [η^2 α^2 0; 0 0 0; 0 0 0]
    M_diag = [0 0 0; η^2 α^2 η^2; 0 0 0]
    M = spzeros(3*T,3*T)
    for i=1:T-1
        M[3*(i-1)+1:3*i,3*(i-1)+1:3*i] += M_diag
        M[3*(i-1)+1:3*i,3*i+1:3*(i+1)] += M_plus
        M[3*i+1:3*(i+1),3*(i-1)+1:3*i] += M_minus
    end

    M[3*(T-1)+1:3*T,3*(T-1)+1:3*T] += M_diag

    return M
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
This function creates the dynamical Bethe-Hessian matrix with parameters 0≤ξ,h≤1, defined in Equation 7
of (Dall'Amico 2020)

Usage
----------
```H = dyn_BH_matrix(AT, ξ, h)```


Entry
----------
* ```AT``` : ```At[t]``` is the sparse representation of the adjacency matrix at time ```t``` (Array{SparseMatrixCSC{Float64,Int64},1})
* ```ξ``` : spatial coupling strength (Float64)
* ```h``` : temporal coupling strength (Float64)

Returns
-------
```H``` : dynamical Bethe-Hessian matrix (nT×nT SparseMatrixCSC{Float64,Int64})
"""
function dyn_BH_matrix(AT::Array{SparseMatrixCSC{Float64,Int64},1}, ξ::Float64, h::Float64)

    T = length(AT) # number of time steps
    n = length(AT[1][1,:]) # number of nodes
    Id = spdiagm(0 => ones(n)) # identity matrix of size n
    dT = [AT[i]*ones(n) for i=1:T] # dT[t] is the degree vector at time t
    DT = [spdiagm(0 => dT[i]) for i=1:T] # DT[t] is the degree matrix at time t
    Zeros = spdiagm(0 => zeros(n))
    HT = [Zeros for i=1:T]
    for i=2:T-1
        HT[i] += (ξ^2*DT[i]-ξ*AT[i])/(1-ξ^2)+(1+h^2)/(1-h^2)*Id   # diagonal block
    end
    HT[1] += (ξ^2*DT[1]-ξ*AT[1])/(1-ξ^2)+ 1/(1-h^2)*Id # boundary term
    HT[T] += (ξ^2*DT[T]-ξ*AT[T])/(1-ξ^2)+ 1/(1-h^2)*Id # boundary term

    H = spdiagm(0 => zeros(n*T)) # initialization of the dynamical BH

    for i=1:T-1
        H[(i-1)*n+1:i*n,(i-1)*n+1:i*n] += HT[i]
        H[(i-1)*n+1:i*n,i*n+1:(i+1)*n] += -h/(1-h^2)*Id
        H[i*n+1:(i+1)*n, (i-1)*n+1:i*n] += -h/(1-h^2)*Id
    end

    H[(T-1)*n+1:n*T,(T-1)*n+1:n*T] += HT[T]

    return H
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


"""
This function computes an approximation of the eigenspace spanned by the eigenvectors associated to
negative eigenvalues of H_{ξ,h} ∈ R^{nT×nT}, according to Algrotithm 2 of (Dall'Amico 2020)

Usage
----------
```X = approximate_embedding(H, verbose; m)```


Entry
----------
* ```H``` : dynamical Bethe-Hessian matrix (nT×nT SparseMatrixCSC{Float64,Int64})
* ```verbose``` : (0, 1, or 2) if 0, nothing is printed. If 1, some information is printed.
    If 2, more information is printed. Default is 1.

Optional input
---------

```m``` : order  of the polynomial approximation (Int64). If ```m``` is not known, an adaptive choice for ```m``` will be  adopted

Returns
----------

```X``` : rectangular matrix containing an approximation of the eigenspace spanned by the eigenvectors
    associated to negative eigenvalues of H (Array{Float64,2})
"""
function approximate_embedding(H::SparseMatrixCSC{Float64,Int64}, verbose; m = nothing)
    n = size(H)[1] # size of matrix H
    Id = spdiagm(0 => ones(n)) # identity matrix
    c = 0

    if verbose >= 1;  print("Computing the smallest eigenvalue of H: λmin. \n"); end
    ## shift H to make it SDP:
    λmin, _ = KrylovKit.eigsolve(H, 1, :SR, tol=2e-5)
    λmin = minimum(λmin)
    if λmin > c - 2e-5
        if verbose >= 1;  print("there is no eigenvalue smaller than ", string(c), " in H! The smallest eigenvalue found is ", string(λmin), "\n"); end
        return 0
    end
    H -= λmin * Id
    # H is now SDP: its first eig is null and all its eigs are >=0 and the number of eigenvalues smaller than c of
    # the original H is equal to the number of eigenvalues smaller than c - λmin in the shifted H

    if verbose >= 1;  print("Computing the largest eigenvalue of H - λmin * Id.\n"); end
    λmax, _, _ = KrylovKit.eigsolve(H, 1, :LR, krylovdim=20, tol=2e-3)
    λmax = (1+2e-3) * maximum(λmax) #to have an extra assurance we have an upper bound

    # compute the polynomial coefficients of the jackson-Chebychev approximation to the ideal low-pass with cut-off c - λmin:
    if m == nothing
        if verbose >= 1;  print("Estimating an adapted polynomial approximation order: "); end
        # estimate a "good" m :
        m_max = 1000
        ch, jch = jackson_cheby_poly_coefficients(c - λmin, λmax, m_max)
        Nv = 3 # number of independent estimations
        signal = randn(n, Nv) # create Nv random Gaussian signals
        sq_sum, order_poly = est_k_cheby(H, ch, signal, λmax) # apply the CH poly of H to each of the coluns of signal and output their norm
        m = Int(median(order_poly))
        k_est = Int(ceil(sum(sq_sum .* (order_poly ./ sum(order_poly)))))
        if verbose >= 1;  print("found m = ", string(m), " (the number of negative eigenvalues of H is estimated at ", string(k_est), ").\n"); end
        ch = ch[1:m]
    else
        ch, jch = jackson_cheby_poly_coefficients(c - λmin, λmax, m)
    end

    #
    Nv = Int(ceil(3*log(n))) # number of independent estimations
    random_matrix = randn(n, Nv) # create Nv random Gaussian signals
    if verbose >= 1;  print("Computing the approximate embedding.\n"); end
    res = cheby_op(H, ch, random_matrix, λmax)

    return res
end



#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
This function computes the approximation of a low pass filter between 0 and λmax on the spectrum of the matrix H ∈ R^{nT×nT}.

Usage
----------
```X = cheby_op(H, ch, random_matrix, λmax)```

Entry
----------
* ```H``` : matrix ```H```  (nT×nT SparseMatrixCSC{Float64,Int64})
* ```ch``` : vector containing the Chebyshev coefficients of the polynomial approximation of the filter (Array{Float64,1})
* ```random_matrix``` : matrix containing in its columns gaussian vector with zero mean ∈ R^{nT}  (Array{Float64,2})
* ```λmax``` : largest eigenvalue of ```H``` (Float64)


Returns
----------

```X``` : rectangular matrix containing an approximation of the eigenspace associated to the ideal low-pass on the spectrum of H
"""
function cheby_op(H::SparseMatrixCSC{Float64,Int64}, ch::Array{Float64,1}, random_matrix::Array{Float64,2}, λmax::Float64)
    N = size(H,1) # size of the matrix  H
    M = size(ch,1) # order of the polynomial approximation

    @assert size(random_matrix,1) == N
    @assert λmax > 0

    λmin = 0.

    Nv = size(random_matrix,2)

    arange = [λmin, λmax]

    a1 = (arange[2] - arange[1]) / 2
    a2 = (arange[2] + arange[1]) / 2

    factor = (2/a1) * (H - a2 * spdiagm(0 => ones(N)))

    res = zeros(N, Nv)
    for i=1:Nv
        twf_old = random_matrix[:,i]
        twf_cur = (H * twf_old - a2 * twf_old) / a1
        r = (0.5 * ch[1]) * twf_old + ch[2] * twf_cur

        for k = 2:M-1 # we do the first 50 rounds in any case
            twf_new = factor * twf_cur - twf_old
            r += ch[k+1] * twf_new
            twf_old = twf_cur
            twf_cur = twf_new
        end
        res[:,i] = r
    end
    return res
end






end # module
