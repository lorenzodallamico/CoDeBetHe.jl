[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://lorenzodallamico.github.io/CoDeBetHe.jl/)


# CoDeBetHe.jl
##### **Co**mmunity **de**tection with the **Bet**he **He**ssian 

This package is an afficient implpementation of the algorithms for spectral community detection  introduced in:
* Dall'Amico, Couillet and Tremblay - *[Revisiting the Bethe-Hessian: improved community detection in sparse heterogeneous graphs](https://lorenzodallamico.github.io/articles/BH19.pdf)* (NeurIPS 2019)
* Dall'Amico, Couillet and Tremblay - *[A unified framework for spectral clustering in sparse graphs](https://lorenzodallamico.github.io/articles/unified_20.pdf)*  (arXiv:2003.09198)
* Dall'Amico, Couillet and Tremblay - *[Community detection in sparse time-evolving graphs with a dynamical Bethe-Hessian](https://lorenzodallamico.github.io/articles/neurips_2020.pdf)* (arXiv:2006.04510)

> If you make use **CoDeBetHe** please consider to cite the above references. 

Beyond the implementation of the algorithms for community reconstruction, the package contains functions to generate synthetic graphs according to the static and dynamic degree corrected stochastic block model, as well as some real datasets.


##### Content of the package

* The directory ```src``` contains the file ```CoDeBetHe.jl``` with the source codes
* The folder ```datasets``` contains```dataset.zip``` (don't forget to unzip the datasets.zip folder before running the demos) with some reals datasets on which aour algortihms can be trun and ```dataset_reference.txt``` with the references of the corresponding datasts. If you are using these datasets for research purpose, please consider to cite the authors of the corresponding dataset.

## Getting Started

These are the basic instructions to use **CoDeBetHe** on you computer

### Installing

- You can install this toolbox by either typing (in the pkg manager)
  '''add https://github.com/lorenzodallamico/CoDeBetHe''
  or cloning the repo locally and typing (in the pkg manager) '''add CoDeBetHe'''

- Don't forget to unzip the real-world graph data in the folder demonstrating the algorithm on real data experiments

### Required packages

**CoDeBetHe** requires the following packages

```
Distributions, LinearAlgebra, DataFrames, StatsBase, IterativeSolvers, Clustering, SparseArrays, KrylovKit, LightGraphs, DelimitedFiles, ParallelKMeans
```

### Usage

To get instructions on how to use the package CoDeBetHe, please refer to the [documentation](https://lorenzodallamico.github.io/CoDeBetHe.jl/) page. There you can find some scripts to easily use the main functions to generate synthetic static and dynamic graphs with communities, to load the datasets inside the ```datasets``` folder and run the community detection algorithms. 

## Authors

[Lorenzo Dall'Amico](https://lorenzodallamico.github.io/)
[Nicolas Tremblay](http://www.gipsa-lab.fr/~nicolas.tremblay/)

## License

This software is released under the GNU AFFERO GENERAL PUBLIC LICENSE (see included file LICENSE)

