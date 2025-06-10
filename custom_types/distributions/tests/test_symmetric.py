
import pytest
from custom_types.distributions.tests.conftest import create_one_var_solutions
from custom_types.distributions.monotonic_distributions import PointSeparationMutation, SampleCountMutation, DistributionShift
from custom_types.distributions.symmetric_distributions import *
from custom_types.distributions.symmetric_bijection import *
from custom_types.distributions._distribution_tools import DistributionInfo

def _linear_return(x, b, slope):
    return (b + slope*x)
def _linear_inverse(y, b, slope):
    return (y - b) / slope
def _linear_bijection(intercept, y_bound, slope):
    x_min = None
    x_max = None
    x_bound = (y_bound - intercept) / slope 
    if intercept < y_bound:
        if slope < 0:
            x_min = x_bound
            x_max = 0
        else:
            x_min = 0
            x_max = x_bound
    else:
        if slope < 0:
            x_min = 0
            x_max = x_bound
        else:
            x_min = x_bound
            x_max = 0
    
    return (partial(_linear_return, b= intercept, slope = slope), 
            partial(_linear_inverse, b= intercept, slope = slope),
            x_min, x_max)

@pytest.fixture(params = ['none', 'include', 'exclude'])
def inclusive_center(request):
    return request.param

@pytest.fixture(params = [(4,4), (4,10), (5,10), (4,11)],ids=lambda v: f"npoints={v}")
def npoints(request):
    return request.param

@pytest.fixture(params = [True, False], ids=lambda v: f"upward_bijection={v}" )
def upward_bijection(request):
    return request.param

@pytest.fixture(params = [True, False], ids=lambda v: f"right_provided={v}" )
def right_provided(request):
    return request.param

@pytest.fixture(params = [True, False], ids=lambda v: f"fixed_width={v}" )
def fixed_width(request):
    return request.param

@pytest.fixture
def symmetric_linear_func(upward_bijection, right_provided):
    """Returns: foward_func, inverse_func, x_min, x_max, y_min, y_max, upperward_func, right_provided"""
    foward_func = None
    inverse_func = None
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    if upward_bijection:
        y_min = 1.0
        y_max = 8.0
        slope = 2.0 if right_provided else -2.0
        foward_func, inverse_func, x_min, x_max = _linear_bijection(
                intercept=y_min, y_bound = y_max, slope = slope
            )
        if right_provided:
            assert foward_func(0.5) == 2
            assert inverse_func(2) == 0.5
            assert x_max == 3.5
        else:
            assert foward_func(-0.5) == 2
            assert inverse_func(2) == -0.5
            assert x_min == -3.5
        
    else:
        y_min = -2.0
        y_max = -1.0
        slope = -2.0 if right_provided else 2.0
        foward_func, inverse_func, x_min, x_max = _linear_bijection(
            intercept = y_max, y_bound = y_min, slope = slope)
        if right_provided:
            assert foward_func(0.25) == -1.5
            assert inverse_func(-1.5) == 0.25
            assert x_max == 0.5
        else:
            assert foward_func(-0.25) == -1.5
            assert inverse_func(-1.5) == -0.25
            assert x_min == -0.5
            
    return (foward_func, inverse_func, 
            x_min, x_max, y_min, y_max, 
            upward_bijection, right_provided)

@pytest.fixture
def symmetric_distribution(symmetric_linear_func, npoints, inclusive_center, fixed_width):
    """Return: SymmetricDistribution object (1 map), y_min, y_max, """
    foward_func, inverse_func, x_min, x_max, y_min, y_max, upward_bijection, right_provided = symmetric_linear_func
    max_width = 2.0*(x_max - x_min)
    min_width = max_width if fixed_width else (x_max - x_min) / 2
    min_points, max_points = npoints
    
    include_extrema = inclusive_center == 'include'
    exclude_extrema = inclusive_center == 'exclude'
    
    sym_bijection = SymmetricBijection(
        forward_function = foward_func,
        inverse_function= inverse_func,
        center_x = 0.0,
        right_side_provided=right_provided,
        min_width = min_width,
        max_width = max_width,
        min_points=min_points,
        max_points=max_points,
        include_global_extrema=include_extrema,
        exclude_global_extrema=exclude_extrema)
    
    set_y_min, set_y_max = sym_bijection.y_bounds
    if right_provided:
        assert not sym_bijection.direction if upward_bijection else sym_bijection.direction
        assert np.isclose(set_y_max, y_max, 1e-7, 1e-8), f"actual y_max {y_max}, set y_max {set_y_max}"
    else:
        assert sym_bijection.direction if upward_bijection else not sym_bijection.direction
        assert np.isclose(set_y_min, y_min, 1e-7, 1e-8), f"actual y_min {y_min}, set y_min {y_max}"

    return SymmetricDistributions([sym_bijection])
        
class TestSymmetricDistributions:
    
    def test_symmetric_bounds(self, symmetric_linear_func, npoints, inclusive_center, fixed_width):
        _, _, x_min, x_max, _, _, _, right_provided= symmetric_linear_func
        max_width = 2.0*(x_max - x_min)
        min_width = max_width if fixed_width else (x_max - x_min) / 2
        min_points, max_points = npoints
        
        include_extrema = inclusive_center == 'include'
        exclude_extrema = inclusive_center == 'exclude'
        
        pb = create_bounds_for_symmetry(
            center_x = 0.0,
            min_width = min_width,
            max_width= max_width,
            min_points = min_points,
            max_points = max_points,
            include_global_extrema=include_extrema,
            exclude_global_extrema=exclude_extrema,
            right_side_provided=right_provided
        )
        assert not pb.max_points > max_points
        if fixed_width and right_provided:
            # print(f"actual points {min_points, max_points}, set_points = {pb.min_points, pb.max_points}")
            assert pb.min_last_point == pb.upper_bound
        elif fixed_width:
            # print(f"actual points {min_points, max_points}, set_points = {pb.min_points, pb.max_points}")
            assert pb.max_first_point == pb.lower_bound
        
        if pb.min_points == pb.max_points or pb.min_separation == pb.max_separation:
            print(f"orig points = {min_points, max_points}: \n {pb!r} \n\n" )
    
    def test_rand(self, symmetric_distribution: SymmetricDistributions):
        output_distrib: DistributionInfo = symmetric_distribution.rand()
        bounds: PointBounds = symmetric_distribution.map_suite[0].point_bounds
        assert output_distrib.output_min_x >= bounds.lower_bound
        assert output_distrib.output_max_x <= bounds.upper_bound
        assert output_distrib.separation >= bounds.min_separation
        assert output_distrib.separation <= bounds.max_separation
        assert output_distrib.num_points >= bounds.min_points
        assert output_distrib.num_points <= bounds.max_points