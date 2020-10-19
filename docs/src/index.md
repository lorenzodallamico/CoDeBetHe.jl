# CoDeBetHe

This module allows to perform **Co**mmunity **De**tection with the **Be**the **He**ssian matrix on sparse unweighted and undirected graphs. The module is composed mainly of two algorithms, one for static community detection, following the results presented in [`A unified framework for spectral clustering in sparse graphs`](https://lorenzodallamico.github.io/articles/unified_20.pdf) and [`Revisiting the Bethe-Hessian: Improved Community Detection in Sparse Heterogeneous Graphs`](https://lorenzodallamico.github.io/articles/BH19.pdf) and one for dynamic community detection, following  [`Community detection in sparse time-evolving graphs with a dynamical Bethe-Hessian`](https://lorenzodallamico.github.io/articles/neurips_2020.pdf).


> If you use CoDeBetHe, please consider to cite the related articles.


This implementations introduce further tricks to make the algorithms more efficent. In order to use CoDeBetHe, open your terminal and type

```
julia
pkg> add https://github.com/lorenzodallamico/CoDeBetHe
```

In the next we will show how to use the main packages for static and dynamic community detection.


```@contents
```


