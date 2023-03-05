# HelmholtzAccCNN.jl

Solution of the Helmholtz equation using CNN preconditioning for acceleration, augomented with shifted Laplacian multigrid.

Code repository for the [paper](https://epubs.siam.org/doi/10.1137/21M1433514) ([arXiv version](https://arxiv.org/abs/2203.11025)).

In this paper, we present a data-driven approach to iteratively solve the discrete heterogeneous Helmholtz equation at high wavenumbers. In our approach, we combine classical iterative solvers with convolutional neural networks (CNNs) to form a preconditioner which is applied within a Krylov solver. For the preconditioner, we use a CNN of type U-Net that operates in conjunction with multigrid ingredients. Two types of preconditioners are proposed 1) U-Net as a coarse grid solver, and 2) U-Net as a deflation operator with shifted Laplacian V-cycles. Following our training scheme and data-augmentation, our CNN preconditioner can generalize over residuals and a relatively general set of wave slowness models. On top of that, we also offer an encoder-solver framework where an "encoder" network generalizes over the medium and sends context vectors to another "solver" network, which generalizes over the right-hand-sides. We show that this option is more robust and efficient than the stand-alone variant. Lastly, we also offer a mini-retraining procedure, to improve the solver after the model is known. This option is beneficial when solving multiple right-hand-sides, like in inverse problems. We demonstrate the efficiency and generalization abilities of our approach on a variety of 2D problems.


If you found the paper useful for your work, please consider citing us:

```bibtex
@article{Azulay2022MultigridAugomented,
author = {Azulay, Yael and Treister, Eran},
title = {Multigrid-Augmented Deep Learning Preconditioners for the Helmholtz Equation},
journal = {SIAM Journal on Scientific Computing},
volume = {0},
number = {0},
pages = {S127-S151},
year = {0},
doi = {10.1137/21M1433514},
URL = {https://doi.org/10.1137/21M1433514},
eprint = {https://doi.org/10.1137/21M1433514},
}
```
