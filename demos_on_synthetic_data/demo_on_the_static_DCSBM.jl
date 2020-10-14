using CoDeBetHe
using Distributions, DelimitedFiles, LinearAlgebra, DataFrames, StatsBase


## Parameter initialization
# These are some lines of codes to initialize the parameters of the DC-SBM

n = floor(Int64,10^(4)) # size of the network (number of nodes)
k = 23 # number of clusters (k)

### vector θ
θ = rand(Uniform(3, 10),n).^3.5
θ = θ./mean(θ)
Φ = mean(θ.^2)

### vector π
var_π = 2
π_v = abs.(rand(Normal(1/k,var_π/(2*k)),k))
π_v = π_v/sum(π_v)
Π = Diagonal(π_v)

### matrix C
c = 10.
c_out = 2.
f = 2/k
C = matrix_C(c_out,c,f,π_v)

ℓ = create_label_vector(n, k, π_v) # create the label vector
A = adjacency_matrix_DCSBM(C,c,ℓ,θ) # create the adjacency matrix of an instance of DC-SBM

verbose = 1; #set to 0 for no verbosity; 1 for some verbosity; 2 for full verbosity
k_prior = nothing; # if you know k in advance, set it here. If not, it is estimated
cluster = community_detection_optimal_BH(A; ℓ = ℓ, k = k_prior, verbose = verbose) # run the community detection algorithm; ℓ is only needed as entry to compute the overlap

## results

printstyled("\n_________________________________________________________________\n"; color = 9)
printstyled("\nThe number of nodes used is: n = ", n; color = 9)
printstyled("\nThe average degree is: c = ", c; color = 9)
printstyled("\nThe value of Φ = ", Φ,2; color = 9)
printstyled("\n"; color = 9)
printstyled("\nThe estimated number of classes is: k_est = ", cluster.k; color = 9)
printstyled("\nThe actual number of classes is: k = ", k; color = 9)
printstyled("\n")
printstyled("\nThe overlap obtained is: ov = ", cluster.overlap; color = 9)
printstyled("\nThe modularity obtained is: mod = ", cluster.modularity; color = 9)
