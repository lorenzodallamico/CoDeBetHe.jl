[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://lorenzodallamico.github.io/CoDeBetHe.jl/)


# CoDeBetHe.jl
##### **Co**mmunity **de**tection with the **Bet**he **He**ssian 

This package is an afficient implementation in Julia language of the algorithms for spectral community detection  introduced in:
* Dall'Amico, Couillet and Tremblay - *[Revisiting the Bethe-Hessian: improved community detection in sparse heterogeneous graphs](https://lorenzodallamico.github.io/articles/BH19.pdf)* (NeurIPS 2019)
* Dall'Amico, Couillet and Tremblay - *[A unified framework for spectral clustering in sparse graphs](https://lorenzodallamico.github.io/articles/unified_20.pdf)*  (JMLR)
* Dall'Amico, Couillet and Tremblay - *[Community detection in sparse time-evolving graphs with a dynamical Bethe-Hessian](https://lorenzodallamico.github.io/articles/neurips_2020.pdf)* (NeurIPS2020)

> If you make use **CoDeBetHe** please consider to cite the above references. 

```
@inproceedings{dall2019revisiting,
  title={Revisiting the Bethe-Hessian: improved community detection in sparse heterogeneous graphs},
  author={Dall'Amico, Lorenzo and Couillet, Romain and Tremblay, Nicolas},
  booktitle={Advances in Neural Information Processing Systems},
  pages={4039--4049},
  year={2019}}
```
```
@article{JMLR:v22:20-261,
  author  = {Lorenzo Dall'Amico and Romain Couillet and Nicolas Tremblay},
  title   = {A Unified Framework for Spectral Clustering in Sparse Graphs},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {217},
  pages   = {1-56},
  url     = {http://jmlr.org/papers/v22/20-261.html}}
```
```
@article{dall2020community,
  title={Community detection in sparse time-evolving graphs with a dynamical Bethe-Hessian},
  author={Dall'Amico, Lorenzo and Couillet, Romain and Tremblay, Nicolas},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}}
```


## Update

In the folder `python` we added two files that allow you to easily use our algorithm in Python as well. They rely on PyJulia and require that Julia is installed on your computer. Recall that the first time that you use the function, it will be particularly slow, unlike typical Python functions.

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

