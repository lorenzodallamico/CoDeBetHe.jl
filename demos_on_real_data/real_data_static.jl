using CoDeBetHe
using Distributions, DelimitedFiles, LinearAlgebra, DataFrames, StatsBase

##

"""
the datasets are called

karate, dolphins, polbooks, football, email, polblogs, tv, fb, power, politicians, Gnutella, vip
"""

el = convert(Array{Int64}, readdlm("datasets/Gnutella.txt")) # upload edge_list

if minimum(el) == 0
    el[:,1] = el[:,1] .+ 1
    el[:,2] = el[:,2] .+ 1
end

fs = vcat(el[:,1],el[:,2]) # symmetrize the edges (ij), (ji)
ss = vcat(el[:,2],el[:,1])

edge_list = hcat(fs,ss) # create edge list
n = length(unique(edge_list)) # find the value of n
A = sparse(edge_list[:,1],edge_list[:,2], ones(length(edge_list[:,1])), n,n) # create sparse adjacency matrix

cluster = community_detection_optimal_BH(A) # run the community detection algorithm



print("\nThe number of nodes used is: n = ", n)
print("\n")
print("\nThe estimated number of classes is: k_est = ", cluster.k)
print("\n")
print("\nThe modularity obtained is: mod = ", cluster.modularity)
print("\n")
