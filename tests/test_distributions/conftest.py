import pytest
from pyomo.environ import Objective
from custom_types.distributions import point_bounds
from numpy import float32, float64


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
    
    
    


