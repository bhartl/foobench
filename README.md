# foobench
A collection of optimization function benchmarks

## Install
Download the github repository, unpack, and navigate into the repository's root folder ``cd foobench`. From within your desired python environment, you may install the repository [optionally in developer mode] via
```bash
pip install [-e] .
```

## Structure
The `foobench` package is structured as follows:
- `foobench.objective` contains a wrapper to treat arbitrary functions as objective functions either for minimization or maximization problems. Objectives can be dumped to and loaded from JSON representations. Thus, they can be easily utilized for large scale benchmarking for optimization algorithms. 
- `foobench.foo` contains several prominent objective functions
- `foobench.plot` contains the plotting functions (WIP)
- `foobench.utils` contains utility functions (WIP)
