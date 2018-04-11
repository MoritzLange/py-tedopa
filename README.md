# py-tedopa
## TEDOPA: Time Evolving Density matrices using Orthogonal Polynomials
This is a Python 3 package for time evolving the states of certain open quantum systems using orthogonal polynomials. It provides:
* Exact mapping of the Hamiltonian of a system, linearly coupled to a continuous bosonic environment, to a one dimensional chain
* Time evolution for matrix product states or operators based on Trotter decomposition
* Functions combining mapping and time evolution, i.e. providing the full TEDOPA

## Installation
Requirements:

* [py-orthpol](https://github.com/moritzlange/py-orthpol), [mpnum](https://github.com/dseuss/mpnum), numpy, scipy, setuptools

The package can be installed by running

    git clone https://github.com/moritzlange/py-tedopa
    cd py-tedopa
    pip install .
on Unix.

## Authors
* [Moritz Lange](https://github.com/moritzlange), Ulm University
* [Ish Dhand](https://github.com/ishdhand), Ulm University

## License 
Distributed under the terms of the BSD 3-Clause License (see [LICENSE](LICENSE)).