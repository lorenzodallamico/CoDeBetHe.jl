using CoDeBetHe
using Distributions, DelimitedFiles, LinearAlgebra, DataFrames, StatsBase


## Parameter initialization
# These are some lines of codes to initialize the parameters of the DC-SBM

n = floor(Int64,10^(3)) # size of the networks (number of nodes)
k = 11 # number of clusters (k)
η = 0.7 # label persistence
T = 50 # number of time-steps

### vector θ

θ = rand(Uniform(3, 10),n).^3
θ = θ./mean(θ)
Φ = mean(θ.^2)

### vector π

var_π = 1
π_v = abs.(rand(Normal(1/k,var_π/(2*k)),k))
π_v = π_v/sum(π_v)
Π = Diagonal(π_v)

### matrix C

c = 8.
c_out = 1.5
f = 1/k
C = matrix_C(c_out,c,f,π_v)

### Then we create the dynamic graph and the successive label vectors ℓ_T

AT, ℓ_T = adjacency_matrix_DDCSBM(T, C, c, η, θ, π_v)

### and perform SC on the dynamical Bethe Hessian:

approx_embedding = false # if true, uses the compressive approximate embedding. Else, full computation of the eigenspace associated to the negative eigenvalues of H
m = nothing # (useful only if approx_embedding == true)
# either set m to a nothing value, and an automatic m will be estimated (usually, the larger n, the larger the required m, the longer the computation)
# or set m to an Int64 if you want to specify the polynomial order used for the approximation
# (as a general rule: the larger m, the more precise the approximate embedding, thus the better the result, but the longer the computation time)
verbose = 1; #set to 0 for no verbosity; 1 for some verbosity
cluster = dynamic_community_detection_BH(AT, η, k; ℓ_T = ℓ_T, approx_embedding = approx_embedding, m = m, verbose=verbose)

## results

printstyled("\n_________________________________________________________________\n"; color = 9)
printstyled("\nThe number of nodes used is: n = ", n; color = 9)
printstyled("\nThe number of time steps considered is T = ", T; color = 9)
printstyled("\nThe value of η = ", η; color = 9)
printstyled("\nThe average degree is: c = ", c; color = 9)
printstyled("\nThe value of Φ = ", Φ,2; color = 9)
printstyled("\nThe number of classes is: k = ", k; color = 9)
printstyled("\n"; color = 9)
if cluster.overlap != "not available"
    printstyled("\nThe mean overlap obtained is: ov = ", mean(cluster.overlap); color = 9)
end
printstyled("\nThe mean modularity obtained is: mod = ", mean(cluster.modularity); color = 9)
printstyled("\nThe self consistent estimate of η is: η = ", mean([(sum(cluster.ℓ[t,:] .== cluster.ℓ[t+1,:])/n-1/k)/(1-1/k) for t=1:T-1]); color = 9)
