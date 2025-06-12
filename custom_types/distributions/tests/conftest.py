import pytest
import numpy as np
import copy
from typing import Literal
from custom_types.distributions import point_bounds, real_bijection
from custom_types.distributions.example_bijections import half_life_bijection
from custom_types.distributions.monotonic_distributions import *
from custom_types.core import *


@pytest.fixture(
    params = [
        (2,None),
        (2,2),
        (100, 100),
        (5, 10000)
    ],
    ids=lambda v: f"cardinality_bounds={v}"
)
def cardinality_bounds(request):
    return request.param

@pytest.fixture(
    params = [
        (-1.0, 1.0),
        (-1001.0, 0.0),
        (-0.0001, 10000),
        (0, 0.0001)
    ], 
    ids=lambda v: f"line_bounds={v}")
def full_bounds(request):
    return request.param

@pytest.fixture(
    params = [
        "min_dist_small", "min_dist_large", "no_min_dist", 
        "fixed_first", "fixed_last"],
    ids=lambda v: f"first_last_bound_type={v}"
)
def first_last_bound_type(request):
    """Describes how to create first point and lower last point bounds"""
    return request.param

@pytest.fixture( params = ['single', 'double'], ids=lambda v: f"precision={v}" )
def float_type(request):
    return request.param

@pytest.fixture
def simple_point_bounds(cardinality_bounds, full_bounds, float_type):
    min_points, max_points = cardinality_bounds
    lb, ub = full_bounds
    
    return point_bounds.PointBounds(
        lb, ub, 
        minimum_points=min_points,
        maximum_points=max_points,
        precision = float_type)

HALF_LIFE_PARAMS = [
    [
        {'y_min': 0.0, 'y_max': 100.0,  'max_y_separation': 0.05, 'T': 4.0, "m": 1.0}, # many possible points
        {'y_min': 0.0, 'y_max': 100.0,  'max_y_separation': 32.0, 'T': 4.0, "m": 1.0}, # limited points
        {'y_min': 0.001, 'y_max': .01, 'max_y_separation': 0.0001, "T": 7.0, "m": 1.0}, # small x separations, lbs, ubs
        {'y_min': -100.0, 'y_max': -99.0, 'max_y_separation': 0.001, "T": 10.0, 'm': 1.0}, # small x separations 2
        {'y_min': -1.0, 'y_max': 10000.0, 'max_y_separation': 5000.0, 'T': 500.0, 'm': 5.0} # small y separations
    ],
    [ "many_points", "limited_points", "small_x_sep_1", "small_x_sep_2", "small_y_large_x" ]   
]

BOUND_OPTIONS = ['fixed_first', 'fixed_last', 'fixed_width_all', 'fixed_width_half', 'strict_points', 'variable', 'none']
COMBINED_OPTIONS = list(('fixed_bound_type', b) for b in BOUND_OPTIONS) + list(('fixed_func', f) for f in range(len(HALF_LIFE_PARAMS[0])))


@pytest.fixture( params = HALF_LIFE_PARAMS[0], ids = HALF_LIFE_PARAMS[1] )
def half_life_func(request):
    return request.param

@pytest.fixture( params = BOUND_OPTIONS, ids = lambda v: f"bound_type={v}" )
def bound_options(request) -> Literal['fixed_first', 'fixed_last', 'fixed_width_all', 'fixed_width_half', 'variable', 'strict_points' 'none']:
    return request.param

def _add_bound_option(bounds: PointBounds, curr_option, x_max, min_x_separation, eps):
    """Assumes lower_bounds == 0"""
    if curr_option == 'fixed_first':
        bounds.set_first_point_upper_bound(0.0)
        
    elif curr_option =='fixed_last':
        bounds.set_last_point_lower_bound(x_max)
        
    elif curr_option == 'variable' or curr_option == 'strict_points':
        
        # print(f"Before bounds set: min_sep {bounds.min_separation}, max_sep {bounds.max_separation}")
        quarter = bounds.upper_bound / bounds.dtype(4)
        mid = bounds.upper_bound / bounds.dtype(2)
        if quarter >= max(min_x_separation, eps):
            bounds.set_first_point_upper_bound(quarter)
            bounds.set_last_point_lower_bound(mid)
        else:
            bounds.set_first_point_upper_bound(mid - 2.0*eps)
            bounds.set_last_point_lower_bound(mid + 2.0*eps)
        
        if curr_option == 'strict_points':
            bounds.set_min_points(bounds.max_points)
            assert bounds.min_points == bounds.max_points

    elif curr_option =='fixed_width_all':
        bounds.set_fixed_width(x_max)
        
    elif curr_option =='fixed_width_half':
        bounds.set_fixed_width(x_max / bounds.dtype(2))
    
    if bounds.min_separation <= 1e-8:
        bounds.set_min_separation(1e-8)

@pytest.fixture
def half_life_bounds(half_life_func, bound_options, x_max):
    """Return:
        forward_func, inverse_func, bounds (PointBounds, y_min, y_max
    """
    forward_func, inverse_func, x_max, min_x_separation, abs_max_points = half_life_bijection(**half_life_func)
    max_points = min(abs_max_points, 100)
    
    bounds = point_bounds.PointBounds(
        0.0, x_max, 
        inclusive_lower=True,
        inclusive_upper = True,
        maximum_points = max_points,
        precision = 'double')
    
    x_max = bounds.dtype(x_max)
    eps = bounds.get_separation_eps()
    
    _add_bound_option(bounds, bound_options, x_max, min_x_separation, eps)
        
    assert bounds.max_points <= min(100,abs_max_points)
    return forward_func, inverse_func, bounds, half_life_func['y_min'], half_life_func['y_max']

@pytest.fixture
def half_life_real_bijection(half_life_bounds):
    forward_func, inverse_func, bounds, _, _ = half_life_bounds
    bijection = real_bijection.RealBijection(
        forward_func,
        bounds,
        inverse_func
    )
    return bijection

@pytest.fixture(params=[True, False], ids=lambda v: "ascending" if v else "descending", )
def distribution_sort(request):
    return request.param

@pytest.fixture(
    params = COMBINED_OPTIONS, # tuples (str, str | int) -> if fixed_func, second elem an index
    ids = lambda v: (f"{v[0]}" + f", {v[1]}") if v[0] == 'fixed_bound_type' else (f"{v[0]}" + f", {HALF_LIFE_PARAMS[1][v[1]]}")
)
def combined_mono_distribution_options(request):
    fixed_str, fixed_val = request.param
    if fixed_str == 'fixed_bound_type':
        return fixed_str, fixed_val
    else:
        return fixed_str, HALF_LIFE_PARAMS[0][fixed_val]

@pytest.fixture
def all_monotonic_distributions(combined_mono_distribution_options, distribution_sort):
    fixed_str, fixed_val = combined_mono_distribution_options
    bijections = []
    if fixed_str == 'fixed_bound_type':
        for bijection_params in HALF_LIFE_PARAMS[0]:
            forward_func, inverse_func, x_max, min_x_separation, abs_max_points = half_life_bijection(**bijection_params)
            max_points = min(abs_max_points, 100)
        
            bounds = point_bounds.PointBounds(
                0.0, x_max, 
                inclusive_lower=True,
                inclusive_upper = True,
                maximum_points = max_points,
                precision = 'double')
            eps = bounds.get_separation_eps()
        
            _add_bound_option(bounds, fixed_val, x_max, min_x_separation, eps)
            assert bounds.max_points <= min(100,abs_max_points)
            bijections.append(real_bijection.RealBijection(
                forward_func,
                bounds,
                inverse_func
            ))
            
    else:
        
        forward_func, inverse_func, x_max, min_x_separation, abs_max_points = half_life_bijection(**fixed_val)
        max_points = min(abs_max_points, 100)
        bounds = point_bounds.PointBounds(
            0.0, x_max, 
            inclusive_lower=True,
            inclusive_upper = True,
            maximum_points = max_points,
            precision = 'double')
        
        eps = bounds.get_separation_eps()
        for bound_option in BOUND_OPTIONS:
            _add_bound_option(bounds, bound_option, x_max, min_x_separation, eps)
            assert bounds.max_points <= min(100,abs_max_points)
            bijections.append(real_bijection.RealBijection(
                forward_func,
                bounds,
                inverse_func
            ))
    
    return MonotonicDistributions(bijections, sort_ascending = distribution_sort)
    
@pytest.fixture( params = {'x_based', 'y_based'}, ids=lambda v: f"distribution_pcx={v}" )
def distribution_pcx_param(request):
    return request.param

@pytest.fixture( params = {'x_based', 'y_based', 'ordinal'}, ids=lambda v: f"distribution_map_variator={v}" )
def distribution_map_variator(request):
    return request.param

@pytest.fixture(params = {'points', 'shift', 'separation'}, ids=lambda v: f"distribution_mutator={v}")
def distribution_mutator(request):
    return request.param

@pytest.fixture
def distribution_pcx(distribution_pcx_param, all_monotonic_distributions):
    y_based = distribution_pcx_param == 'y_based' 
    all_monotonic_distributions.local_variator = DistributionBoundsPCX(0.999, y_based_pcx=y_based)
    all_monotonic_distributions.do_evolution = True
    return all_monotonic_distributions

@pytest.fixture
def distribution_map_crossover(distribution_map_variator, all_monotonic_distributions):
    all_monotonic_distributions.ordinal_maps = distribution_map_variator == 'ordinal'
    y_based = distribution_map_variator == 'y_based'
    all_monotonic_distributions.local_variator = FixedMapCrossover(0.999, y_based_map_crossover=y_based)
    all_monotonic_distributions.do_evolution = True
    return all_monotonic_distributions

@pytest.fixture
def distribution_map_mutation(distribution_map_variator, all_monotonic_distributions):
    all_monotonic_distributions.ordinal_maps = distribution_map_variator == 'ordinal'
    y_based = distribution_map_variator == 'y_based'
    all_monotonic_distributions.local_variator = FixedMapConversion(0.999, y_based_map_conversion=y_based)
    all_monotonic_distributions.do_mutation = True
    return all_monotonic_distributions

@pytest.fixture
def distribution_bound_mutation(distribution_mutator, all_monotonic_distributions):
    local_variator = None
    if distribution_mutator == 'points':
        local_variator = SampleCountMutation(0.999)
    elif distribution_mutator == 'shift':
        local_variator = DistributionShift(0.999)
    else: # PointSeparationMutation
        local_variator = PointSeparationMutation(0.999 , separation_alpha=2, separation_beta=2)
    
    all_monotonic_distributions.local_variator = local_variator
    all_monotonic_distributions.do_mutation = True
    return all_monotonic_distributions
    
@pytest.fixture(params = [True, False], ids=lambda v: f"deepcopy={v}")
def deepcopy_parents(request):
    return request.param

@pytest.fixture( params=[(3, 4), (2,1), (2,2)],
                ids=lambda v: f"nsolutions={v}" )
def nsolutions_crossover(request):
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
                sol.evaluated = True
                offspring_solutions.append(sol)
                     
        return parent_solutions, offspring_solutions, copy_indices
    


    
    
        


