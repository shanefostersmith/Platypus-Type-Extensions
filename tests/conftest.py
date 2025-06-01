import pytest
import numpy as np
from platypus import Problem, Solution
from custom_types.core import CustomType, LocalVariator, LocalMutator
import copy

@pytest.fixture(
    params=[1, 2, 5],
    ids=lambda v: f"noffspring={v}"
)
def noffspring(request):
    """
    Returns an noffspring count (in 1,2,5)
    """
    return request.param

@pytest.fixture(
    params=[1, 2, 5],
    ids=lambda v: f"nparents={v}"
)
def nparents(request):
    """
    Returns an nparents count (in 1,2,5)
    """
    return request.param

@pytest.fixture(params=[1, 2, 10], ids=lambda s: f"size={s}")
def vector_size(request):
    return request.param

@pytest.fixture(params=["zeros", "random"], ids=lambda k: k)
def vector_kind(request):
    return request.param

@pytest.fixture
def np_normalized_vector32(vector_size, vector_kind):
    size = vector_size
    kind = vector_kind

    if kind == "zeros":
        return np.zeros(size, dtype=np.float32)
    else:
        rng = np.random.default_rng(0)
        out = rng.random(size, dtype=np.float32)
        return out

@pytest.fixture(params=[1, 2, 5], ids=lambda n: f"nrows={n}")
def nrows(request): # nparents
    return request.param

@pytest.fixture(params=[1, 2, 5], ids=lambda n: f"ncols={n}")
def ncols(request): # nvars
    return request.param 

@pytest.fixture(params=[True,False],ids=lambda v: f"zero_reference={v}")
def zero_reference(request):
    "whether the reference (last) row is all zeros"
    return request.param

@pytest.fixture(params=[True,False], ids=lambda v: f"zero_column={v}")
def zero_column(request):
    "whether the first column is all zeros"
    return request

@pytest.fixture
def np_normalized_matrix32(nrows, ncols, zero_reference, zero_column):
    """
    Builds float32 matrix
    """
    rng = np.random.default_rng(0)
    rand_matrix = np.empty(shape=(nrows,ncols), dtype = np.float32, order = 'C')
    rng.random(dtype=np.float32, out=rand_matrix)
    if zero_reference:
        rand_matrix[-1, :] = 0
    if zero_column:
        rand_matrix[:, 0] = 0
    return rand_matrix

@pytest.fixture(
    params=['variate', 'mutate'],
    ids=lambda v: f"variation_type={v}"
)
def single_variation_type(request):
    return request.param

@pytest.fixture(params = [True, False], ids=lambda v: f"deepcopy={v}")
def deepcopy_parents(request):
    return request.param

def simpleFunc(vars: list):
    """A Problem function for testing types. 
    
    Takes some number of vars, applies random objective scores.
    Assumes no constraints, and 1 objective"""
    
    return [np.random.uniform()], []

def unconstrainedProblem(*vars):
    """Create a Problem with any number of decision variables, one objective and no contraints"""
    nvars = len(vars)
    problem = Problem(nvars, nobjs = 1, function=simpleFunc)
    for i, v in enumerate(vars):
        problem.types[i] = v
    return problem

def create_one_var_solutions(custom_type: CustomType, nparents = 2, noffspring = 1, deepcopy = False) -> tuple[list[Solution], list[Solution]] | tuple[None, Solution]:
    """Create Solution objects for a Variator or Mutator
    
    If deepcopy is True, deepcopies parents in order to create offspring solutions. 
        The last parent index is deepcopied multiple times if nparents < noffspring

    Returns
        tuple[list[Solution], list[Solution]] | tuple[None, Solution]: 
        - If custom_type contains a LocalMutator, returns (None, Solution), 
        - Otherwise, returns (list[Solution], list[Solution]) where the number of Solution is given by 'nparents' and 'noffspring'
    """
    problem = unconstrainedProblem(custom_type)
    if issubclass(type(custom_type.local_variator), LocalMutator):
        offspring_sol = Solution(problem)
        offspring_sol.variables[0] = custom_type.rand()
        return None, offspring_sol, [None]
    else:
        parent_solutions = [Solution(problem) for _ in range(nparents)]
        copy_indices = [None for _ in range(noffspring)]
        offspring_solutions = None
        for sol in parent_solutions:
            sol.variables[0] = custom_type.rand()
        if not deepcopy:
            offspring_solutions = [Solution(problem) for _ in range(noffspring)]
            for sol in offspring_solutions:
                sol.variables[0] = custom_type.rand()
        else:
            offspring_solutions = []
            for i in range(noffspring):
                par_idx = min(nparents - 1, i)
                copy_indices[i] = par_idx
                offspring_solutions.append(copy.deepcopy(parent_solutions[par_idx]))
                    
        return parent_solutions, offspring_solutions, copy_indices

