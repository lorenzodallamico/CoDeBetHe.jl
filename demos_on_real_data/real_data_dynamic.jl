using CoDeBetHe
using Distributions, DelimitedFiles, LinearAlgebra, DataFrames, StatsBase, SparseArrays

##

# Load the data

edge_list = convert(Array{Int64}, readdlm("datasets/el_primary.dat"))
index = convert(Array{Int64}, readdlm("datasets/index_primary.dat"))
tt = convert(Array{Int64}, readdlm("datasets/times_primary.dat"))

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

η = 0.55

k = 10
cluster = dynamic_community_detection_BH(AT, η, k; approx_embedding = true)



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
printstyled("\nThe modularity obtained is: mod = ", mean(cluster.modularity); color = 9)
printstyled("\nThe self consistent estimate of η is: η = ", mean([(sum(cluster.ℓ[t,:] .== cluster.ℓ[t+1,:])/n-1/k)/(1-1/k) for t=1:T-1]); color = 9)
