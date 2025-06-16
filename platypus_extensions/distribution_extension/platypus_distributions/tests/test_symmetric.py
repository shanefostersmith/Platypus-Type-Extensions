
import pytest
from platypus_extensions.core import LocalCompoundMutator
from .conftest import create_one_var_solutions
from ..monotonic_distributions import PointSeparationMutation, SampleCountMutation
from ..symmetric_distributions import *
from ..symmetric_bijection import *
from .._distribution_tools import DistributionInfo

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

@pytest.fixture(params = ['none', 'include', 'exclude'], ids=lambda v: f"inclusion_type={v}")
def inclusive_center(request):
    return request.param

@pytest.fixture(params = [(4,4), (4,10), (5,10), (10,11)],ids=lambda v: f"npoints={v}")
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
    else:
        y_min = -2.0
        y_max = -1.0
        slope = -2.0 if right_provided else 2.0
        foward_func, inverse_func, x_min, x_max = _linear_bijection(
            intercept = y_max, y_bound = y_min, slope = slope)
           
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
    
    assert sym_bijection._x_point_bounds.max_points < max_points*2
    
    set_y_min, set_y_max = sym_bijection.y_bounds
    if right_provided:
        assert not sym_bijection.direction if upward_bijection else sym_bijection.direction
        assert np.isclose(set_y_max, y_max, 1e-7, 1e-8), f"actual y_max {y_max}, set y_max {set_y_max}"
    else:
        assert sym_bijection.direction if upward_bijection else not sym_bijection.direction
        assert np.isclose(set_y_min, y_min, 1e-7, 1e-8), f"actual y_min {y_min}, set y_min {y_max}"

    return SymmetricDistributions([sym_bijection])
        
class TestSymmetricDistributions:
    rtol = 1e-8
    atol = 1e-9
    def test_symmetric_bounds(self, symmetric_linear_func, npoints, inclusive_center, fixed_width):
        forward_func, inverse_func, x_min, x_max, _, _, _, right_provided= symmetric_linear_func
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
            right_side_provided=right_provided,
            create_bounds_state=True
        )
        
        sym_bij = SymmetricBijection(
            forward_func,
            inverse_function=inverse_func,
            center_x = 0,
            min_width = min_width,
            max_width = max_width,
            min_points=min_points,
            max_points=max_points,
            right_side_provided = right_provided,
            include_global_extrema=include_extrema,
            exclude_global_extrema=exclude_extrema,
            point_bounds=pb,
        )
        
        full_min_separation, full_max_separation = sym_bij._full_separation_bounds
        if not full_min_separation == full_max_separation:
            assert full_min_separation < pb.max_separation or np.isclose(
                full_min_separation, pb.max_separation, self.rtol, self.atol)
            
            assert full_max_separation >= pb.min_separation or np.isclose(
                full_max_separation, pb.min_separation, self.rtol, self.atol)
            
    def test_rand(self, symmetric_distribution: SymmetricDistributions):
        output_distrib: DistributionInfo = symmetric_distribution.rand()
        bounds: PointBounds = symmetric_distribution.map_suite[0].point_bounds
        assert output_distrib.output_min_x >= bounds.lower_bound
        assert output_distrib.output_max_x <= bounds.upper_bound
        assert output_distrib.separation >= bounds.min_separation
        assert output_distrib.separation <= bounds.max_separation
        assert output_distrib.num_points >= bounds.min_points
        assert output_distrib.num_points <= bounds.max_points
    
    def test_pre_adjustment(self,symmetric_distribution: SymmetricDistributions):
        output_distrib: DistributionInfo = symmetric_distribution.rand()
        sym_bij: SymmetricBijection = symmetric_distribution.map_suite[0]
        bounds: PointBounds = sym_bij.point_bounds
        full_min_separation, full_max_separation = sym_bij._full_separation_bounds
        full_min_points, full_max_points = sym_bij._full_point_bounds
        
        # set extremes
        if full_min_separation - bounds.min_separation  > 0:
            output_distrib.separation = bounds.min_separation
            output_distrib.num_points = bounds.max_points
        elif bounds.max_separation - full_max_separation > 0:
            output_distrib.separation = bounds.max_separation
            output_distrib.num_points = bounds.min_points
        elif 2*bounds.min_points < full_min_points:
            output_distrib.num_points = bounds.min_points
        elif 2*bounds.max_points > full_max_points:
            output_distrib.num_points = bounds.max_points
        
        output_start, output_points, signed_separation, include_center = sym_bij._pre_distribution(
            output_distrib.output_min_x,
            output_distrib.num_points,
            output_distrib.separation
        )
        if not sym_bij.right_side_provided:
            assert signed_separation < 0 
        
        output_separation = abs(signed_separation)
        half_width = bounds.dtype(output_points - 1)*output_separation
        full_width = 2.0 * half_width 
        if not include_center:
            full_width += output_separation
        
        full_min_width, full_max_width = sym_bij._full_width_bounds
        assert full_min_points <= 2*output_points - include_center <= full_max_points
        assert full_min_width < full_width < full_max_width or (
            np.isclose(full_min_width, full_width, self.rtol, self.atol) or
            np.isclose(full_max_width, full_width, self.rtol, self.atol)
        )
        assert full_min_separation < output_separation < full_max_separation or (
            np.isclose(full_min_separation, output_separation, self.rtol, self.atol) or
            np.isclose(full_max_separation, output_separation, self.rtol, self.atol)
        )
        if include_center:
            assert output_start == sym_bij.center_x
            assert not sym_bij.exclude_global_extrema
        else:
            assert not sym_bij.include_global_extrema
            half_signed_separation = signed_separation / 2.0
            assert np.isclose(output_start, sym_bij.center_x + half_signed_separation, self.rtol, self.atol)

    def test_decode(self, symmetric_distribution: SymmetricDistributions, request):
        output_distrib: DistributionInfo = symmetric_distribution.rand()
        sym_bij: SymmetricBijection = symmetric_distribution.map_suite[0]

        decoded = sym_bij.create_distribution(
            start_x= output_distrib.output_min_x, 
            num_points=output_distrib.num_points, 
            separation= output_distrib.separation
        )
        d_npoints = len(decoded)
        if sym_bij.include_global_extrema:
            assert d_npoints % 2 == 1
        if sym_bij.exclude_global_extrema:
            assert d_npoints % 2 == 0
            
        y_left, _ = sym_bij.left_y_bounds
        _, y_right = sym_bij.right_y_bounds
        y_center = y_left if sym_bij.right_side_provided else y_right  
        assert np.isclose(sym_bij.fixed_inverse_map(y_center), sym_bij.center_x, self.rtol, self.atol)
        
        approx_separation = None
        mid_low_idx = None
        if d_npoints % 2:
            mid_low_idx  = d_npoints // 2
            y_center_decoded = decoded[mid_low_idx]
            assert y_center_decoded == y_center
            assert np.all(decoded[:mid_low_idx][::-1] == decoded[mid_low_idx + 1:])
        else:
            mid_low_idx = d_npoints // 2 - 1
            if sym_bij.right_side_provided:
                mid_high_idx = mid_low_idx + 1
                after_center = decoded[mid_high_idx]
                x_after = sym_bij.fixed_inverse_map(after_center)
                x_after2 = sym_bij.fixed_inverse_map(decoded[mid_high_idx + 1])
                approx_separation = abs(x_after - x_after2)
                assert np.isclose(sym_bij.fixed_forward_map(sym_bij.center_x + approx_separation / 2.0), after_center, self.rtol, self.atol)
            else:
                before_center = decoded[mid_low_idx]
                x_before = sym_bij.fixed_inverse_map(before_center)
                x_before2 = sym_bij.fixed_inverse_map(decoded[mid_low_idx - 1])
                approx_separation = abs(x_before - x_before2)
                assert np.isclose(sym_bij.fixed_forward_map(sym_bij.center_x - approx_separation / 2.0), before_center, self.rtol, self.atol)
            assert np.all(decoded[:mid_low_idx+1][::-1] == decoded[mid_low_idx+1:])
        
        upward = request.getfixturevalue("upward_bijection") # upward -> V shape, not upward -> upside down V
        if upward:
            assert np.all(np.diff(decoded[:mid_low_idx + 1]) < 0)
        else:
            assert np.all(np.diff(decoded[:mid_low_idx + 1]) > 0)
        
        if approx_separation is not None:
            full_min_separation, full_max_separation = sym_bij._full_separation_bounds
            assert full_min_separation < approx_separation < full_max_separation or (
                np.isclose(full_min_separation, approx_separation, self.rtol, self.atol) or
                np.isclose(full_max_separation, approx_separation, self.rtol, self.atol)
            )
        if sym_bij.right_side_provided:
            x_max = sym_bij.fixed_inverse_map(decoded[-1])
            assert sym_bij.point_bounds.min_last_point <= x_max <= sym_bij.point_bounds.upper_bound or (
                np.isclose(x_max, sym_bij.point_bounds.min_last_point, self.rtol, self.atol) or
                np.isclose(x_max, sym_bij.point_bounds.upper_bound, self.rtol, self.atol)
            )
        else:
            x_min = sym_bij.fixed_inverse_map(decoded[0])
            assert sym_bij.point_bounds.lower_bound <= x_min <= sym_bij.point_bounds.max_first_point or (
                np.isclose(x_min, sym_bij.point_bounds.max_first_point, self.rtol, self.atol) or
                np.isclose(x_min, sym_bij.point_bounds.lower_bound, self.rtol, self.atol)
            )
    def test_encode(self, symmetric_distribution: SymmetricDistributions):
        output_distrib: DistributionInfo = symmetric_distribution.rand()
        sym_bij: SymmetricBijection = symmetric_distribution.map_suite[0]

        decoded = sym_bij.create_distribution(
            start_x= output_distrib.output_min_x, 
            num_points=output_distrib.num_points, 
            separation= output_distrib.separation
        )
        encoded: DistributionInfo = symmetric_distribution.encode((decoded, output_distrib.map_index))
        bounds: PointBounds = symmetric_distribution.map_suite[0].point_bounds
        assert encoded.output_min_x >= bounds.lower_bound
        assert encoded.output_max_x <= bounds.upper_bound
        assert encoded.separation >= bounds.min_separation
        assert encoded.separation <= bounds.max_separation
        assert encoded.num_points >= bounds.min_points
        assert encoded.num_points <= bounds.max_points
    
    def test_with_mutator(self, symmetric_distribution: SymmetricDistributions):
        lm = LocalCompoundMutator([PointSeparationMutation(0.999), SampleCountMutation(0.999)])
        symmetric_distribution.local_variator = lm
        symmetric_distribution.do_mutation = True
        bounds = symmetric_distribution.map_suite[0].point_bounds
        _, offspring_sol, _ = create_one_var_solutions(symmetric_distribution, nparents = 0)
        lm.mutate(symmetric_distribution, offspring_sol, variable_index=0)
        new_distrib: DistributionInfo  = offspring_sol.variables[0]
        assert new_distrib.output_min_x >= bounds.lower_bound
        assert new_distrib.output_max_x <= bounds.upper_bound
        assert new_distrib.separation >= bounds.min_separation
        assert new_distrib.separation <= bounds.max_separation
        assert new_distrib.num_points >= bounds.min_points
        assert new_distrib.num_points <= bounds.max_points