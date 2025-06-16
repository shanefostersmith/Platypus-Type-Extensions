# Platypus-Type-Extensions

This library introduces new variable types for multi-objective optimization, and associated crossover and mutation strategies. 
It also includes a flexible framework to for defining new variable types and optimizing mixed-type problems with a single "global" variator.
All variable types and strategies are fully compatible with the Platypus MOO framework.

## New Optimization Variables

- **Floating Point Array**
    - Optimize a Numpy array of floats with Numba-accelerated strategies 
    - (PCX, Differential Evolution, Polynomial Mutation, etc.)
- **Integer Array**
    - Optimize a Numpy array of integers with Numba-accelerated strategies 
    - (Integer Crossover, Binary Swap, Bit Flip, etc.)
- **SetPartition**:
    - Optimize a subset of elements partitioned into bins
- **WeightedSet**:
    - Optimize a subset of elements and the weight associated with each active feature
- **Monotonic and Symmetric Distributions**
    - Optimize a mapping of evenly-spaced 'x' values to 'y' values with any user-defined distribution function
    - Define the dynamic bounds of the output distributions (bounds on the x and y domains, width, cardinality and separation between points)
- **Categories**
    - Optimize categorical variables
- **SteppedRange**
    - Optimize a number within a range of step values
- **RealList**
    - Optimize a number within any sequence of numbers

## Installation

Basic Install (no extras)
```
    pip install git+https://github.com/shanefostersmith/platypus-type-extensions.git
```

With the distribution variable types:
```
    pip install git+https://github.com/shanefostersmith/platypus-type-extensions.git#egg=platypus-type-extensions[distributions]
```

With the tests:
```
    pip install git+https://github.com/shanefostersmith/platypus-type-extensions.git#egg=platypus-type-extensions[tests]
```

With all extras (distributions + tests):
```
    pip install git+https://github.com/shanefostersmith/platypus-type-extensions.git#egg=platypus-type-extensions[full]
```

Or, from a local clone:
```
    git clone https://github.com/shanefostersmith/platypus-type-extensions.git
    cd platypus-type-extensions

    # Core library only
    pip install -e .

    # Core + distributions
    pip install -e ".[distributions]"

    # Core + tests
    pip install -e ".[tests]"

    # All extras
    pip install -e ".[full]"
```

## Requirements

### Core Dependencies
-  Python >= 3.9
- `platypus-opt>=1.4.1`  
- `numpy>=1.26.0`  
- `numba>=0.60.0`  
- `scipy>=1.10.0`  

> Numpy, Numba and Scipy are used for the vectorized / accelerated float, integer and linear algebra methods 

### Optional Dependencies
The distribution extra also requires:
- `pyomo>=6.8.2`

The test extra also requires:
- `pytest>=7.0.0`
- `pytest-mock>=3.10.0`

## Usage

The main difference between this library and the Platypus library is the distinction between "local" variators and "global" variators.

- A `GlobalEvolution` directs the crossover and mutation of all optimization variables, and may define non-uniform mutation behavior, staggered convergence, etc.

- A `LocalVariator` or `LocalMutator` defines a crossover or mutation strategy for specific variable type(s)
    - Each optimization variable, including variables of the *same* type, can be assigned a different `LocalVariator`
    - A `LocalVariator` also defines compatible arities for its `evolve()` method (ie. the number of input parent and offspring solutions)
    - A `LocalMutator` is a special type of `LocalVariator` that mutates one offspring solution at a time without using parent solutions
        - Its `evolve()` method can still accept any number of offspring solutions

- This library also provides compound operators for `LocalVariator`:
    - `LocalGAOperator`: Combine a `LocalVariator` and a `LocalMutator`
    - `LocalCompoundMutator`: Compound any number of `LocalMutator`
    - `LocalSequentialOperator`: Compound any number of `LocalVariator` and `LocalMutator`

- The `CustomType`, `LocalVariator`, and `GlobalEvolution` class docs provide information on how to build your own optimization variable types and evolution strategies

### An Example

Optimizing three variables of two different types with Platypus's NSGAII algorithm

```python

    from platypus import Problem, NSGAII
    from platypus_extensions import (
        GeneralGlobalEvolution, 
        SteppedRange, SteppedRangeCrossover, StepMutation, 
        RealList, RealListPM)

    # LocalVariators
    step_variator = SteppedRangeCrossover(step_crossover_rate = 0.2)
    step_mutator = StepMutation(mutation_probability = 0.1)
    real_list_mutator = RealListPM(mutation_probability = 0.2)

    # Optimization variables
    step_variable1 = SteppedRange(
        lower_bound = 0, 
        upper_bound = 1, 
        step_value = 0.1, 
        local_variator = step_variator
    )
    step_variable2 = SteppedRange(
        lower_bound = -1, 
        upper_bound = 0,
        step_value = 0.2, 
        local_variator = step_variator, 
        local_mutator = step_mutator
    ) 
    real_list_variable = RealList(
        real_list = [-0.1, 0.25, 0.5], 
        local_mutator = real_list_mutator
    )

    # Problem definition
    def eval_func(vars):
        return [(var[0] + var[1]) / var[2]]
    problem = Problem(nvars = 3, nobjs = 1, function = eval_func)
    problem.types[:] = [step_variable1, step_variable2, real_list_variable]

    # GlobalEvolution and Algorithm
    global_variator = GeneralGlobalEvolution(nparents = 4, noffspring = 2)
    algorithm = NSGAII(problem = problem, variator = global_variator)
    algorithm.run(10000)
```
- `GeneralGlobalEvolution` executes the optimization variables' LocalVariators during each step of the algorithm
    - You should ensure that the `nparents` and `noffspring` parameters are compatible with the LocalVariators' supported arities
    - LocalMutators are compatible with any `nparents` and `noffspring`

- Providing both a `LocalVariator` and a `LocalMutator` to an optimization variable automatically creates a `LocalGAOperator`

- LocalVariators may be provided to optimization variables at initialization or after initialization (and may be altered or replaced at any time)


## License
As an extension of the Platypus multi-objective optimization library, this library inherits the same GNU General Public License

> Hadka, D. (2024). Platypus: A Framework for Evolutionary Computing in Python (Version 1.4.1) [Computer software].  Retrieved from https<span>://</span>github.com/Project-Platypus/Platypus.