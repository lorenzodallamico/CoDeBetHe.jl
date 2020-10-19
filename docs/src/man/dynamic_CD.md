# Dynamic community detection


## Create a synthetic graphs


The package CoDeBetHe allows to generate graphs with communities according to the dynamical degree corrected stochastic block model (D-DCSBM). 

Let ``T`` be the number of different snapshots that compose the dynamical graph ``\{\mathcal{G}_t\}_{t=1,\dots,T}``. Suppose that each graph is composed by ``n`` nodes and ``k`` communities, with label vector ``\bm{\ell}_t \in \{1,\dots,k\}^n`` and the adjacency matrix ``A^{(t)} \in \{0,1\}^{n\times n}`` for ``1\leq t\leq T``. \
The label vector ``\bm{\ell}_{t = 1}`` is initialized so that ``\pi_a\cdot n`` nodes have label ``a``. The labels are then updated for ``2 \leq t \leq T`` according to the Markov process, where ``\ell_{i_t}`` denote the label of node ``i`` at time ``t``.
```math
\ell_{i_t} = \begin{cases}
\ell_{i_{t-1}}~{\rm w.p.}~ \eta \\
a ~ {\rm w.p.}~ (1-\eta)\pi_a,~ a \in \{1,\dots k\}
\end{cases}
```

Once the vector label is created, each adjacency matrix ``A^{(t)}`` is drawn independently at random according to the degree-corrected stochastic block model (see ```Static community detection```.
The following lines of codes allow to generate a a graph sequence according to the D-DCSBM. First we initialize the parameters of the DCSBM

> For a more precise description of the DCSBM, please refer to the ```Static community detection``` section.

```julia
using CoDeBetHe
using Distributions, DelimitedFiles, LinearAlgebra, DataFrames, StatsBase


## Parameter initialization

n = floor(Int64,10^(3.5)) # size of the networks (number of nodes)
k = 10 # number of clusters (k)
η = 0.7 # label persistence
T = 6 # number of time-steps

### vector θ

θ = rand(Uniform(3, 10),n).^3
θ = θ./mean(θ)
Φ = mean(θ.^2)

### vector π

var_π = 1
π_v = abs.(rand(Normal(1/k,var_π/(2*k)),k))
π_v = π_v/sum(π_v)

### matrix C

c = 8. # expected average degree
c_out = 1.5 # expected off diagonal-elements of C
f = 1/k # fluctuation of off diagonal elements
C = matrix_C(c_out,c,f,π_v) # creation of the matrix C
``` 

We then create the sequences ``\{\bm{\ell}_t\}_{t=1,\dots,T}`` and ``\{A^{(t)}\}_{t = 1,\dots,T}``

```julia
AT, ℓ_T = adjacency_matrix_DDCSBM(T, C, c, η, θ, π_v)
```

## Load a synthetic data-set


At [github.com/lorenzodallamico/CoDeBetHe](https://github.com/lorenzodallamico/CoDeBetHe), you can find the data-set needed to run the following example. The data are taken from the [SocioPattern](http://www.sociopatterns.org/) project. 

!!! note

    If using this dataset, please give reference to the followings articles:
    * Mitigation of infectious disease at school: targeted class closure vs school closure,
    * High-Resolution Measurements of Face-to-Face Contact Patterns in a Primary School, PLOS ONE 6(8): e23176 (2011)

```julia
# Load the data

edge_list = convert(Array{Int64}, readdlm("datasets/el_primary.dat")) # this loads the edge list
index = convert(Array{Int64}, readdlm("datasets/index_primary.dat")) # this list is to identifies the identities corresponding to each node 
tt = convert(Array{Int64}, readdlm("datasets/times_primary.dat")) # this loads the time at which an edges was present

id = findall(index .== 1)
index = [id[i][1] for i=1:length(id)]

# Generate the temporal graph

T = maximum(tt)
n = maximum(unique(edge_list))+1
AT = [spdiagm(0 => zeros(n)) for i=1:T]

edge_list[:,1] = edge_list[:,1] .+ 1
edge_list[:,2] = edge_list[:,2] .+ 1

for t=1:T
    idx = Array(findall(tt .== t))
    idx = [idx[i][1] for i=1:length(idx)]
    el = edge_list[idx,:]
    AT[t] = sign.(sparse(el[:,1],el[:,2], ones(length(el[:,1])), n,n)) # create sparse adjacency matrix
    AT[t] = AT[t][index,index]
    AT[t] = AT[t] + AT[t]'
end

n = length(index)
T = findmin([sum(AT[t]) for t=1:T])[2]-2
AT = AT[1:T]
```

## Infer the community structure from ``\{A^{(t)}\}_{t = 1,\dots,T}``

We will show the basic usage of the function [`dynamic_community_detection_BH`](@ref), applied to the SocioPattern network. For a more specific use of the outputs, please refer to the documentation of [`dynamic_community_detection_BH`](@ref).

```@example
using CoDeBetHe, Distributions, DelimitedFiles, LinearAlgebra, DataFrames, StatsBase, SparseArrays, Plots # hide


edge_list = convert(Array{Int64}, readdlm("/home/lorenzo/CoDeBetHe/docs/data_for_doc/el_primary.dat")) # hide
index = convert(Array{Int64}, readdlm("/home/lorenzo/CoDeBetHe/docs/data_for_doc/index_primary.dat")) # hide
tt = convert(Array{Int64}, readdlm("/home/lorenzo/CoDeBetHe/docs/data_for_doc/times_primary.dat")) # hide

id = findall(index .== 1) # hide
index = [id[i][1] for i=1:length(id)] # hide
T = maximum(tt) # hide
n = maximum(unique(edge_list))+1 # hide
AT = [spdiagm(0 => zeros(n)) for i=1:T] # hide

edge_list[:,1] = edge_list[:,1] .+ 1 # hide
edge_list[:,2] = edge_list[:,2] .+ 1 # hide

for t=1:T # hide
    idx = Array(findall(tt .== t)) # hide
    idx = [idx[i][1] for i=1:length(idx)] # hide
    el = edge_list[idx,:] # hide
    AT[t] = sign.(sparse(el[:,1],el[:,2], ones(length(el[:,1])), n,n)) # hide
    AT[t] = AT[t][index,index] # hide
    AT[t] = AT[t] + AT[t]' # hide
end # hide

n = length(index) # hide
T = findmin([sum(AT[t]) for t=1:T])[2]-2 # hide
AT = AT[1:T] # hide

η = 0.55 # chose the value of η
k = 10 # set the number of communities 

"""Optional inputs"""

approx_embedding = false # if true, uses the compressive approximate embedding. 
# Else, full computation of the eigenspace associated to the negative eigenvalues of H

m = nothing # (useful only if approx_embedding == true); either set m to a nothing value,
# and an automatic m will be estimated (usually, the larger n, the larger the required m, 
# the longer the computation) or set m to an Int64 if you want to specify the polynomial 
# order used for the approximation (as a general rule: the larger m, the more precise the 
# approximate embedding, thus the better the result, but the longer the computation time)

verbose = 1; #set to 0 for no verbosity; 1 for some verbosity


cluster = dynamic_community_detection_BH(AT, η, k; approx_embedding = approx_embedding, m = m, verbose=verbose)

t_v = Array(0:T-1)*15/60 .+ 8.5 # hide
plot(t_v, cluster.modularity, marker = :dot, color = :orange, linewidth = 3, xlabel = "Time", ylabel = "Modularity", label = "") # hide
```
