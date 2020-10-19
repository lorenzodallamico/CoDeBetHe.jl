# Static community detection


## Create a synthetic graphs


The package CoDeBetHe allows to generate graphs with communities according to the degree corrected stochastic block model (DC-SBM). 

Let ``\mathcal{G}`` be a graph with ``n`` nodes and ``k`` communities. 
Let ``\bm{\ell} \in \{1,\dots,k\}^n`` be the label vector and ``A \in \{0,1\}^{n\times n}`` the adjacency matrix whose entry ``A_{ij}=1`` if and only if ``(ij)`` is an edge of ``\mathcal{G}``. \
The vector ``\bm{\theta} \in \mathbb{R}^n``, satisfying ``\frac{1}{n}\sum_i\theta_i = 1`` and ``\frac{1}{n}\sum_i\theta_i^2 = \Phi``, for some ``\Phi \in \mathbb{R}``. The expected degree of the node ``i`` will be proportional to ``\theta_i`` and therefore the vector ``\bm{\theta}`` can be used to reproduce an arbitrary degree distribution in the graph ``\mathcal{G}``.\
The matrix ``C\in\mathbb{R}^{k\times k}`` then contains at its entry ``C_{ab}`` the affinity between class ``a`` and class ``b``. 

The entries of the adjacency matrix ``A`` are generated independently at random according to

```math
\mathbb{P}(A_{ij} = 1|\ell_i,\ell_j) = \theta_i\theta_j \frac{C_{\ell_i,\ell_j}}{n}
```

The following lines of codes allow to generate a graph with communities according to the DC-SBM


```julia
using CoDeBetHe
using Distributions, DelimitedFiles, LinearAlgebra, DataFrames, StatsBase

n = floor(Int64,10^(4)) # size of the network (number of nodes)
k = 10 # number of clusters (k)

### vector θ
θ = rand(Uniform(3, 10),n).^3.5 # with this we induce a power law degree distribution
θ = θ./mean(θ) # we impose that \sum_i \theta_i = n
```

!!! note

    The definition of the vector ``θ`` should be made according to the degree distribution distribution one whishes to have. In this example we would obtain a power law degree degree distribution, while for a homogeneous degree distribution (recovering the classical stochastic block model), one should simply set ``\theta_i = 1,~\forall~i``. The normalization line should not be changed, instead.

We now proceed choosing the size of each class. To do so we introduce a vector ``\bm{\pi} \in \mathbb{R}^k`` whose entry ``\pi_a`` indicates the fraction of nodes that are in class ``a, \left(\sum_{a = 1}^k \pi_a = 1\right)``. We then define matrix ``\Pi = {\rm diag}(\bm{\pi})``. Here is an example of how to build the vector ``\bm{\pi}``, using gaussian random variables, centered in ``1/k``.


```julia
### vector π
π_v = abs.(rand(Normal(1/k,1/(2*k)),k)) # we take the abs, because π_v[i] > 0 for all i
π_v = π_v/sum(π_v) # Normalize the vector (do not change this line)

```

Finally, we create the matrix ``C``  with the function [`matrix_C`](@ref). The matrix ``C`` is created so that all the rows of the matrix ``C\Pi`` sum up to ``c``, the expected average degree of the network. This conditions implies that the expected average degree is independent of the class: otherwise, by simply looking at the degree distribution one could infer the community structure.\
Further degrees of freedom in the definition of  ``C`` are added through the the parameter ```f``` which is such that the off-diagonal elements are drown from a gaussian distribution with average ```c_out``` and hence are not equal. If one wants all the off-diagonal elements to be equal, simply set ```f = 0```. The following lines create the matrix ``C``.

```julia
### matrix C
c = 10. # average degree
c_out = 2. # average value of off-diagonal terms
f = 2/k # fluctuation of off-diagonal terms
C = matrix_C(c_out,c,f,π_v) # matrix C

```
Given the vectors ``\bm{\pi}, \bm{\theta}`` and the matrix ``C``, we then generate the ground truth vector ``\bm{\ell}`` and the adjacency matrix ``A``	

```julia
ℓ = create_label_vector(n, k, π_v) # create the label vector
A = adjacency_matrix_DCSBM(C,c,ℓ,θ) # create the adjacency matrix of an instance of DC-SBM
```

!!! note

    The matrix ```A``` is stored in sparse format, through its edge list representation. Moving to a dense representation would drastically increase the computational cost.


The following plot shows a toy example of the output of [`adjacency_matrix_DCSBM`](@ref) on a small network with large average degree and four communities of different sizes.

```@example 
using CoDeBetHe, LinearAlgebra, Distributions, Plots # hide

n = floor(Int64,10^(2.5)) # hide
k = 4 # hide

θ = rand(Uniform(3, 10),n).^0 # hide
θ = θ./mean(θ) # hide

π_v = Array([0.1,0.4,0.2,0.3]) # hide
Π = Diagonal(π_v) # hide

c = 100. # hide
c_out = 20. # hide
C = matrix_C(c_out,c,0.,π_v) # hide

ℓ = create_label_vector(n, k, π_v) # hide
A = adjacency_matrix_DCSBM(C,c,ℓ,θ) # hide

heatmap(Array(A), c =cgrad([:white, :blue])) # hide
```

## Load a real graphs 

At [https://github.com/lorenzodallamico/CoDeBetHe](https://github.com/lorenzodallamico/CoDeBetHe), at the directory ```demos_on_real_data``` one can find some real datasets on which our algorithm can tested. Once ```dataset.zip``` has been unzipped, run the following commands to upload one of the networks


```julia
using CoDeBetHe
using Distributions, DelimitedFiles, LinearAlgebra, DataFrames, StatsBase

"""
the datasets are called
karate, dolphins, polbooks, football, email, polblogs, tv, fb, power, politicians, Gnutella, vip
"""

el = convert(Array{Int64}, readdlm("datasets/Gnutella.txt")) # upload edge_list: here you should be putting the name of the dataset you wish to upload

# we let for all networks, the label i.d. range from 1 to n (not from 0 to n-1)

if minimum(el) == 0
    el[:,1] = el[:,1] .+ 1
    el[:,2] = el[:,2] .+ 1
end

fs = vcat(el[:,1],el[:,2]) # symmetrize the edges (ij), (ji)
ss = vcat(el[:,2],el[:,1])

edge_list = hcat(fs,ss) # create edge list
n = length(unique(edge_list)) # find the value of n
A = sparse(edge_list[:,1],edge_list[:,2], ones(length(edge_list[:,1])), n,n) # create sparse adjacency matrix
```

!!! note

    The references for the datasets are
    * Karate : Wayne W Zachary. An information flow model for conflict and fission in small groups.Journalof anthropological research, 33(4):452–473, 1977.
    * Dolphins : David Lusseau, Karsten Schneider, Oliver J Boisseau, Patti Haase, Elisabeth Slooten, andSteve M Dawson.   The bottlenose dolphin community of doubtful sound features a largeproportion of long-lasting associations.Behavioral Ecology and Sociobiology, 54(4):396–405,2003.
    * Polbooks : www.orgnet.com
    * Football : Michelle Girvan and Mark EJ Newman. Community structure in social and biological networks.Proceedings of the national academy of sciences, 99(12):7821–7826, 2002.
    * Polblogs: Lada A Adamic and Natalie Glance. The political blogosphere and the 2004 us election:divided they blog. In Proceedings of the 3rd international workshop on Link discovery,pages 36–43. ACM, 2005.

    All other datasets are taken from the Stanford Network Analysis Project website page http://snap.stanford.edu/
    * Jure Leskovec and Andrej Krevl. SNAP Datasets: Stanford large network dataset collection.http://snap.stanford.edu/data, June 2014.


## Infer the community structure from ``A``

Once we have the matrix ```A```, we can run our Algorithm for community detection. This is done through the function [`community_detection_optimal_BH`](@ref) which is an efficient implementation of Algorithm 2 of [A unified framework for spectral clustering in sparse
graphs](https://lorenzodallamico.github.io/articles/unified_20.pdf). Below you can find the basic use of this function (run on Political blogs) with the typical output of the function. For more details see the documentation of [`community_detection_optimal_BH`](@ref).


```@example
using CoDeBetHe # hide
using Distributions, DelimitedFiles, LinearAlgebra, DataFrames, StatsBase, SparseArrays # hide


el = convert(Array{Int64}, readdlm("https://github.com/lorenzodallamico/CoDeBetHe/tree/main/docs/data_for_doc/polblogs.txt")) # hide


if minimum(el) == 0 # hide
    el[:,1] = el[:,1] .+ 1 # hide
    el[:,2] = el[:,2] .+ 1 # hide
end # hide

fs = vcat(el[:,1],el[:,2]) # hide
ss = vcat(el[:,2],el[:,1]) # hide

edge_list = hcat(fs,ss) # hide
n = length(unique(edge_list)) # hide
A = sparse(edge_list[:,1],edge_list[:,2], ones(length(edge_list[:,1])), n,n) # hide


verbose = 1; #set to 0 for no verbosity; 1 for some verbosity; 2 for full verbosity
k_prior = 2; # if you know k in advance, set it here. If not, it is estimated
cluster = community_detection_optimal_BH(A; k = k_prior, verbose = verbose) # run the community detection algorithm

printstyled("\nThe modularity obtained is: mod = ", cluster.modularity; color = 9)
```

For the use of the outputs of the function [`community_detection_optimal_BH`](@ref), please refer to the documentation page.



