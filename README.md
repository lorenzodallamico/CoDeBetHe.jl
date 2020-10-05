# CoDeBetHe
##### **Co**mmunity **de**tection with the **Bet**he **He**ssian


This package is an afficient implpementation of the algorithms for spectral community detection  introduced in:
* Dall'Amico, Couillet and Tremblay - *[Revisiting the Bethe-Hessian: improved community detection in sparse heterogeneous graphs](https://lorenzodallamico.github.io/articles/BH19.pdf)* (NeurIPS 2019)
* Dall'Amico, Couillet and Tremblay - *[A unified framework for spectral clustering in sparse graphs](https://lorenzodallamico.github.io/articles/unified_20.pdf)*  (arXiv:2003.09198)
* Dall'Amico, Couillet and Tremblay - *[Community detection in sparse time-evolving graphs with a dynamical Bethe-Hessian](https://lorenzodallamico.github.io/articles/neurips_2020.pdf)* (arXiv:2006.04510)

> If you make use **CoDeBetHe** please consider to cite the above references. 

Beyond the implementation of the algorithms for community reconstruction, the package contains functions to generate synthetic graphs according to the static and dynamic degree corrected stochastic block model, as well as some real datasets.


##### Content of the package

* The directory ```src``` contains the file ```CoDeBetHe.jl``` with the source codes
* The folder ```demos_on_synthetic_data``` contains two demo files ```.jl``` that display the basic usage of the package in order to create a synthetic (static or dynamic) graph with communities and subsequently run the community detection algorithm.
* The folder ```demos_on_real_data``` shows the use of the algorithms on real datasets, contained in ```dataset.zip``` (don't forget to unzip the datasets.zip folder before running the demos). The reference of each of the datasets is specified in the file ```dataset_reference.txt```. If you are using these datasets for research purpose, please consider to cite the authors of the corresponding dataset.

## Getting Started

These are the basic instructions to use **CoDeBetHe** on you computer

### Installing

complete...

### Required packages

**CoDeBetHe** requires the following packages

```
Distributions, LinearAlgebra, DataFrames, StatsBase, IterativeSolvers, Clustering, SparseArrays, KrylovKit, LightGraphs, DelimitedFiles, ParallelKMeans
```

## Authors

[Lorenzo Dall'Amico](https://lorenzodallamico.github.io/)
[Nicolas Tremblay](http://www.gipsa-lab.fr/~nicolas.tremblay/)

## License

?

