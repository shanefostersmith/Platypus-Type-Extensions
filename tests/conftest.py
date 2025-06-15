import pytest
import numpy as np
import copy
from typing import Literal
from platypus import Problem, Solution
from platypus_extensions.core import CustomType, LocalVariator, LocalMutator
from platypus_extensions.global_evolutions.general_global_evolution import GeneralGlobalEvolution

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

@pytest.fixture( params = [(5, 100.0), (100, 1e10), (int(1e4), 1e10), (int(1e5), 1e5), (int(2.5e5), 1.1)], ids = lambda v: f"ranges=vector_length: {v[0]}, max_number {v[1]}" )
def ranges_type(request):
    return request.param

@pytest.fixture(
    params = [(5, 100.0), (1000, 1.0e4), (int(1e4), 1.0e10), (int(5e4), 5.0), (int(2e5), 5.0)], ids = lambda v: f"ranges=vector_length: {v[0]}, max_number {v[1]}" #, (int(1e5), 1e5)
)
def np32_ranges_DE(request):
    vector_length, max_val = request.param
    matrix = np.zeros((vector_length, 2), dtype = np.float32)
    rng = np.random.default_rng(0)
    half_max = max_val / 2
    matrix[:,1] = (half_max * rng.random(size = vector_length, dtype = np.float32)) + half_max
    orig = np.full(vector_length, half_max / 2, dtype = np.float32)
    p1 = np.full(vector_length, half_max / 3, dtype = np.float32)
    p2 = np.full(vector_length, half_max / 4, dtype = np.float32)
    p3 = np.full(vector_length, half_max / 2.5, dtype = np.float32)
    return matrix, orig, p1, p2, p3

@pytest.fixture
def np64_ranges(ranges_type) -> tuple[np.ndarray, np.ndarray]:
    """Returns: ranges (2d), values {1d}"""
    vector_length, max_val = ranges_type
    matrix = np.zeros((vector_length, 2), dtype = np.float64)
    rng = np.random.default_rng(0)
    half_max = max_val / 2
    matrix[:,1] = (half_max * rng.random(size = vector_length, dtype = np.float64)) + half_max
    values = rng.random(size = vector_length, dtype = np.float64) * half_max
    return matrix, values

@pytest.fixture(params = [(5, 5), (10, 1000), (1000, 10), (int(2e4), 10), (10, int(2e4))], ids = lambda v: f"ranges=2d_range: parents: {v[0]}, vectors {v[1]}")
def np64_2D_ranges(request) -> tuple[np.ndarray, np.ndarray]:
    """Returns: ranges (2d), values {1d}"""
    rows, vars = request.param
    matrix = np.zeros((vars, 2), dtype = np.float64)
    rng = np.random.default_rng(0)
    half = 0.5 if rows > 5 else 50
    matrix[:,1] = (half * rng.random(size = vars, dtype = np.float64)) + half
    values = rng.random(size = (rows, vars), dtype = np.float64) * half
    return matrix, values

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
    
    return [np.random.uniform()]

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
        custom_type.do_mutation = True
        offspring_sol = Solution(problem)
        offspring_sol.variables[0] = custom_type.rand()
        offspring_sol.evaluated = True
        return None, offspring_sol, [None]
    else:
        custom_type.do_evolution = True
        parent_solutions = [Solution(problem) for _ in range(nparents)]
        copy_indices = [None for _ in range(noffspring)]
        
        offspring_solutions = None
        for sol in parent_solutions:
            sol.variables[0] = custom_type.rand()
            sol.evaluated = True
            
        if not deepcopy:
            offspring_solutions = [Solution(problem) for _ in range(noffspring)]
            for sol in offspring_solutions:
                sol.variables[0] = custom_type.rand()
                sol.evaluated = True
        else:
            offspring_solutions = []
            for i in range(noffspring):
                par_idx = min(nparents - 1, i)
                copy_indices[i] = par_idx
                sol = copy.deepcopy(parent_solutions[par_idx])
                offspring_solutions.append(sol)
                    
        return parent_solutions, offspring_solutions, copy_indices
    
def create_multi_var_solutions(
    *custom_types,
    nsolutions = 2, 
    problem = None):
    
    if problem is None:
        problem = unconstrainedProblem(*custom_types)
    else:
        custom_types = [problem.types[i] for i in range(problem.nvars)]
    
    out = []
    for _ in range(nsolutions):
        solution = Solution(problem)
        solution.evaluated = True
        for custom_type in custom_types:
            for j in range(problem.nvars):
                solution.variables[j] = custom_type.rand()
            if isinstance(custom_type, LocalVariator):
                if issubclass(type(custom_type.local_variator), LocalMutator):
                    custom_type.do_mutation = True
                else:
                    custom_type.do_evolution = True
        out.append(solution)
        
    return out

def create_basic_global_evolution(
    arity, offspring, 
    copy_method: int | Literal['sample', 'rand'] = 'rand'):
    
    return GeneralGlobalEvolution(arity, offspring, copy_method)
    
