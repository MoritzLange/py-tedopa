# py-tedopa

## TEDOPA: Time Evolving Density matrices using Orthogonal Polynomials

[![Documentation Status](https://readthedocs.org/projects/py-tedopa/badge/?version=latest)](http://py-tedopa.readthedocs.org/en/latest/?badge=latest)

This is a Python 3 package for simulating the time-evolution of quantum systems via matrix product states and operators.

Time evolution of one-dimensional quantum systems under local Hamiltonians is provided via the tMPS algorithm using second- and fourth-order Trotter Suzuki decomposition. Hamiltonian evolution of matrix product state, matrix product operators and purified matrix product states is supported.

Time evolution of open quantum systems is implemented using TEDOPA (Time Evolving Density matrices using Orthogonal Polynomials Algorithm) as proposed and developed by [Prior et al 2010](http://link.aps.org/doi/10.1103/PhysRevLett.105.050404), [Chin et al 2010](http://aip.scitation.org/doi/10.1063/1.3490188) and [Rosenbach 2015](https://oparu.uni-ulm.de/xmlui/123456789/3945). TEDOPA enables open quantum simulation of open quantum systems such as spin-boson models.

TEDOPA relies on a two-step approach to simulate quantum systems that a coupled linearly to their continuous environment via a given spectral density: first, the star-shaped system-environment interaction is mapped exactly to a semi-infinite one-dimensional chain with the system at one end of the chain; This chain is then simulated using standard matrix-product-states (MPS) methods.


## Installation
Requirements:

* [py-orthpol](https://github.com/moritzlange/py-orthpol)
* [mpnum](https://github.com/dseuss/mpnum)
* numpy, scipy, setuptools

The package can be installed by running

    git clone https://github.com/moritzlange/py-tedopa
    cd py-tedopa
    pip install .

on Unix-based systems.

## Authors
* [Moritz Lange](https://github.com/moritzlange), Ulm University
* [Ish Dhand](http://ishdhand.me), Ulm University

## License
Distributed under the terms of the BSD 3-Clause License (see [LICENSE](LICENSE)).
