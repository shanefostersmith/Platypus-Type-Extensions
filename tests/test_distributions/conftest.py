import pytest
import numpy as np
from pyomo.environ import Objective
from custom_types.distributions import point_bounds, real_bijection
from custom_types.distributions.ex_bijection_funcs import half_life_bijection
from typing import Literal


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

@pytest.fixture(
    params = [
        {'y_min': 0.0, 'y_max': 100.0,  'max_y_separation': 0.05, 'T': 4.0, "m": 1.0}, # many possible points
        {'y_min': 0.0, 'y_max': 100.0,  'max_y_separation': 32.0, 'T': 4.0, "m": 1.0}, # limited points
        {'y_min': 0.001, 'y_max': .01, 'max_y_separation': 0.0001, "T": 7.0, "m": 1.0}, # small x separations, lbs, ubs
        {'y_min': -100.0, 'y_max': -99.0, 'max_y_separation': 0.001, "T": 10.0, 'm': 1.0}, # small x separations 2
        {'y_min': -1.0, 'y_max': 10000.0, 'max_y_separation': 5000.0, 'T': 500.0, 'm': 5.0} # small y separations
    ],
    ids = [
        "many_points",
        "limited_points",
        "small_x_sep_1",
        "small_x_sep_2",
        "small_y_large_x"
    ]
)
def half_life_func(request):
    return request.param


@pytest.fixture(
    params = ['fixed_first', 'fixed_last', 'fixed_width_all', 'fixed_width_half', 'strict_points', 'variable', 'none'],
    ids = lambda v: f"min_distance_type={v}"
)
def bound_options(request) -> Literal['fixed_first', 'fixed_last', 'fixed_width_all', 'fixed_width_half', 'variable', 'strict_points' 'none']:
    return request.param

@pytest.fixture
def half_life_bounds(half_life_func, bound_options):
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
    
    if bound_options == 'fixed_first':
        bounds.set_first_point_upper_bound(0.0)
        
    elif bound_options =='fixed_last':
        bounds.set_last_point_lower_bound(x_max)
        
    elif bound_options == 'variable' or bound_options == 'strict_points':
        quarter = bounds.upper_bound / bounds.dtype(4)
        mid = bounds.upper_bound / bounds.dtype(2)
        print(f"mid: {mid}, quarter = {quarter}")
        if quarter >= max(min_x_separation, eps):
            print("here good")
            bounds.set_first_point_upper_bound(quarter)
            assert np.isclose(bounds.max_first_point, quarter, 1e-7, 1e-8)
            bounds.set_last_point_lower_bound(mid)
        else:
            bounds.set_first_point_upper_bound(mid - 2.0*eps)
            bounds.set_last_point_lower_bound(mid + 2.0*eps)
        if bound_options == 'strict_points':
            bounds.set_min_points(bounds.max_points)
            assert bounds.min_points == bounds.max_points
            
    elif bound_options =='fixed_width_all':
        bounds.set_fixed_width(x_max)
        
    elif bound_options =='fixed_width_half':
        bounds.set_fixed_width(x_max / bounds.dtype(2))
        
    assert bounds.max_points <= min(100,abs_max_points)
    print(f"bounds: {bounds!r}")
    print(f"bound_type: {bound_options}\n")
    return forward_func, inverse_func, bounds, half_life_func['y_min'], half_life_func['y_max']


@pytest.fixture
def half_life_real_bijection(half_life_bounds):
    forward_func, inverse_func, bounds, _, _ = half_life_bounds
    bijection = real_bijection.RealBijection(
        forward_func,
        bounds,
        inverse_func
    )
    return bijection, _, _
    
    
    
    
        


