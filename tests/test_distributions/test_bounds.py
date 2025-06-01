import pytest
import pyomo as pyo
from custom_types.distributions.point_bounds import *
# from custom_types.distributions.point_bounds import *
from custom_types.distributions import _bounds_tools
from tests.test_distributions.conftest import *
from pyomo.util.infeasible import find_infeasible_constraints, find_infeasible_bounds
# from pyomo.util
from pyomo.opt import SolverFactory, TerminationCondition
from math import floor

class TestPointBounds:

    def test_init(self, simple_point_bounds: PointBounds):
        orig_lb, orig_ub = simple_point_bounds.get_full_bounds()

        bounds_view = simple_point_bounds.create_bounds_state()
        _bounds_tools._cascade_from_global(bounds_view)
        simple_point_bounds._apply_state(_bounds_tools.CascadePriority.GLOBAL, bounds_view)
        new_lb, new_ub = simple_point_bounds.get_full_bounds()
        
        assert orig_lb == new_lb, "lower‐bounds array changed"
        assert orig_ub == new_ub, "upper‐bounds array changed"
        # print(f"max_points: {simple_point_bounds.max_points}")
        
        for constr, body_value, infeasible in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}. Type {infeasible}")
        print(f"Bounds: {simple_point_bounds!r}")
    
    def test_cardinality_set(self, simple_point_bounds: PointBounds):
        orig_min_points = simple_point_bounds.min_points
        orig_max_points = simple_point_bounds.max_points
        orig_lb, orig_ub = simple_point_bounds.get_full_bounds()
        bound_width = orig_ub - orig_lb
        
        if np.isinf(orig_max_points):
            new_max_points = floor((bound_width / 1e-10) + 1.0)
            simple_point_bounds.set_max_points(new_max_points)
            assert not np.isinf(simple_point_bounds.max_points)
            for constr, body_value, _ in find_infeasible_constraints(simple_point_bounds.model):
                raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}")
            simple_point_bounds.set_min_points(2)
        elif orig_min_points > 2:
            simple_point_bounds.set_max_points(orig_min_points - 1)
            assert simple_point_bounds.min_points == simple_point_bounds.max_points
            for constr, body_value, _ in find_infeasible_constraints(simple_point_bounds.model):
                raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}")
            simple_point_bounds.set_min_points(2)
        else:
            simple_point_bounds.set_min_points(orig_max_points + 1)
            for constr, body_value, _ in find_infeasible_constraints(simple_point_bounds.model):
                raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}")
            simple_point_bounds.set_max_points(2)
            
        new_lb, new_ub = simple_point_bounds.get_full_bounds()
        assert orig_lb == new_lb and orig_ub == new_ub
        assert simple_point_bounds.max_separation * (simple_point_bounds.min_points - 1) <= bound_width
        
        for constr, body_value, _ in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}")
    
    def test_first_last_bounds_set(self, simple_point_bounds: PointBounds, first_last_bound_type):
        orig_max_points = simple_point_bounds.max_points
        orig_lb, orig_ub = simple_point_bounds.get_full_bounds()
        bound_width = orig_ub - orig_lb
        mid_point = (orig_ub + orig_lb) / simple_point_bounds.dtype(2.0)
        rtol = 1e-7 if simple_point_bounds.dtype == np.float64 else 1e-6
        atol = 1e-8 if simple_point_bounds.dtype == np.float64 else 1e-7
        
        if first_last_bound_type == "min_dist_small":
            simple_point_bounds.set_first_point_upper_bound(np.nextafter(mid_point, -np.inf, dtype=simple_point_bounds.dtype))
            simple_point_bounds.set_last_point_lower_bound(np.nextafter(mid_point,  np.inf, dtype=simple_point_bounds.dtype))
            assert simple_point_bounds.min_last_point - simple_point_bounds.max_first_point > 0
            
        elif first_last_bound_type == "min_dist_large":
            simple_point_bounds.set_first_point_upper_bound(orig_lb + 2*np.finfo(simple_point_bounds.dtype).eps)
            simple_point_bounds.set_last_point_lower_bound(orig_ub - 2*np.finfo(simple_point_bounds.dtype).eps)
                
        elif first_last_bound_type == "no_min_dist":
            simple_point_bounds.set_first_point_upper_bound(mid_point + np.finfo(simple_point_bounds.dtype).eps)
            simple_point_bounds.set_last_point_lower_bound(mid_point - np.finfo(simple_point_bounds.dtype).eps)
            
        elif first_last_bound_type == "fixed_first":
            simple_point_bounds.set_first_point_upper_bound(orig_lb)
            simple_point_bounds.set_last_point_lower_bound(mid_point)
            simple_point_bounds.set_lower_bound(mid_point)
            assert simple_point_bounds.lower_bound == simple_point_bounds.max_first_point
            assert simple_point_bounds.lower_bound < simple_point_bounds.min_last_point

        else: #fixed_last
            simple_point_bounds.set_last_point_lower_bound(orig_ub)
            simple_point_bounds.set_first_point_upper_bound(mid_point)
            simple_point_bounds.set_upper_bound(mid_point)
            assert simple_point_bounds.upper_bound == simple_point_bounds.min_last_point
            assert simple_point_bounds.upper_bound > simple_point_bounds.max_first_point

        assert 0 < simple_point_bounds.min_separation <= simple_point_bounds.max_separation
        assert 2 <= simple_point_bounds.min_points <= simple_point_bounds.max_points
        # assert simple_point_bounds.max_separation * (simple_point_bounds.min_points - 1) <= bound_width
        
        if not np.isinf(orig_max_points):
            assert not np.isinf(simple_point_bounds.max_points)
            point_min_width = simple_point_bounds.min_separation * (simple_point_bounds.max_points - 1)
            true_min_width = simple_point_bounds.min_last_point - simple_point_bounds.max_first_point
            assert (point_min_width >= true_min_width or np.isclose(point_min_width, true_min_width, rtol = rtol, atol=atol) ),  f"bounds: {simple_point_bounds!r}"
        
        if np.isinf(orig_max_points):
            simple_point_bounds.set_max_points(floor((bound_width / 1e-10) + 1.0))
        simple_point_bounds.set_first_point_upper_bound(None)
        simple_point_bounds.set_last_point_lower_bound(None)
        
        assert 0 < simple_point_bounds.min_separation <= simple_point_bounds.max_separation
        assert 2 <= simple_point_bounds.min_points <= simple_point_bounds.max_points
        # assert simple_point_bounds.max_separation * (simple_point_bounds.min_points - 1) <= bound_width
        
        point_min_width = simple_point_bounds.min_separation * (simple_point_bounds.max_points - 1)
        true_min_width = simple_point_bounds.min_last_point - simple_point_bounds.max_first_point
        assert (point_min_width >= true_min_width or np.isclose(point_min_width, true_min_width, rtol = rtol, atol=atol) ), f"bounds: {simple_point_bounds!r}"
        
        for constr, body_value, _ in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}")
        
    def test_separation_set(self, simple_point_bounds: PointBounds):
        orig_lb, orig_ub = simple_point_bounds.get_full_bounds()
        bound_width = orig_ub - orig_lb
        orig_max_points = simple_point_bounds.max_points
        if np.isinf(orig_max_points):
            simple_point_bounds.set_max_points(floor((bound_width / 1e-10) + 1.0))
            
        mid_point = (orig_ub + orig_lb) / simple_point_bounds.dtype(2.0)
        rtol = 1e-8 if simple_point_bounds.dtype == np.float64 else 1e-6
        atol = 1e-9 if simple_point_bounds.dtype == np.float64 else 1e-7
        
        true_min_sep, _ = simple_point_bounds.get_conditional_separation_bounds(orig_max_points)
        simple_point_bounds.set_min_separation(0.95 * true_min_sep)
        assert simple_point_bounds.max_separation * (simple_point_bounds.min_points - 1) <= bound_width
        for constr, body_value, infeasible in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}. Type {infeasible}")
        
        simple_point_bounds.set_first_point_upper_bound(np.nextafter(mid_point, -np.inf, dtype=simple_point_bounds.dtype))
        simple_point_bounds.set_last_point_lower_bound(np.nextafter(mid_point,  np.inf, dtype=simple_point_bounds.dtype))
        simple_point_bounds.set_first_point_upper_bound(orig_lb + 2*np.finfo(simple_point_bounds.dtype).eps)
        simple_point_bounds.set_last_point_lower_bound(orig_ub - 2*np.finfo(simple_point_bounds.dtype).eps)
        for constr, body_value, infeasible in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}. Type {infeasible}")
        
        _, true_max_sep = simple_point_bounds.get_conditional_separation_bounds(simple_point_bounds.min_points)
        simple_point_bounds.set_max_separation(1.05 * true_max_sep)
        new_min_width = simple_point_bounds.min_separation * (simple_point_bounds.max_points - 1)
        
        assert (new_min_width > simple_point_bounds.min_last_point - simple_point_bounds.max_first_point or 
                np.isclose(new_min_width, simple_point_bounds.min_last_point - simple_point_bounds.max_first_point, rtol = rtol, atol = atol))
        assert simple_point_bounds.max_separation * (simple_point_bounds.min_points - 1) <= bound_width
        for constr, body_value, infeasible in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}. Type {infeasible}")
        
        if not simple_point_bounds.max_separation == simple_point_bounds.max_separation:
            simple_point_bounds.set_min_separation(simple_point_bounds.max_separation)
        else:
            simple_point_bounds.set_min_separation(bound_width)
        assert (new_min_width > simple_point_bounds.min_last_point - simple_point_bounds.max_first_point or 
                np.isclose(new_min_width, simple_point_bounds.min_last_point - simple_point_bounds.max_first_point, rtol = rtol, atol = atol))
        assert simple_point_bounds.max_separation * (simple_point_bounds.min_points - 1) <= bound_width
        
        for constr, body_value, infeasible in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}. Type {infeasible}")
    
    def test_fixed_width(self, simple_point_bounds: PointBounds):
        orig_lb, orig_ub = simple_point_bounds.get_full_bounds()
        bound_width = orig_ub - orig_lb
        orig_max_points = simple_point_bounds.max_points
        dtype = simple_point_bounds.dtype
        if np.isinf(orig_max_points):
            simple_point_bounds.set_max_points(floor((bound_width / 1e-10) + 1.0))
        # print(f"Bounds OG: \n {simple_point_bounds!r}\n")
            
        simple_point_bounds.set_fixed_width(bound_width - 2*np.spacing(bound_width))
        assert not simple_point_bounds.fixed_width > bound_width
        dist_from_lb = simple_point_bounds.max_first_point - simple_point_bounds.lower_bound
        dist_from_ub = simple_point_bounds.upper_bound - simple_point_bounds.min_last_point
        assert(dist_from_lb >= 0 and dist_from_lb == dist_from_ub)
        for constr, body_value, infeasible in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}. Type {infeasible}")
        # print(f"Bounds 1: \n {simple_point_bounds!r}\n")
        
        simple_point_bounds.set_lower_bound(simple_point_bounds.lower_bound + 2.0*np.spacing(simple_point_bounds.lower_bound))
        dist_from_lb = simple_point_bounds.max_first_point - simple_point_bounds.lower_bound
        dist_from_ub = simple_point_bounds.upper_bound - simple_point_bounds.min_last_point  
        assert(dist_from_lb >= 0 and dist_from_lb == dist_from_ub)
        for constr, body_value, infeasible in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}. Type {infeasible}")
        # print(f"Bounds 2: \n {simple_point_bounds!r}\n")
        
        simple_point_bounds.set_upper_bound(simple_point_bounds.upper_bound - 4.0*np.finfo(simple_point_bounds.dtype).eps)
        dist_from_lb = simple_point_bounds.max_first_point - simple_point_bounds.lower_bound
        dist_from_ub = simple_point_bounds.upper_bound - simple_point_bounds.min_last_point

        assert(dist_from_lb >= 0 and dist_from_lb == dist_from_ub)
        for constr, body_value, infeasible in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}. Type {infeasible}")
        # print(f"Bounds 3: \n {simple_point_bounds!r}\n")
        
        
        eps = simple_point_bounds.get_separation_eps()
        simple_point_bounds.set_fixed_width(eps)
        abs_min_dist = dtype(simple_point_bounds.min_points - 1) * eps
        dist_from_lb = simple_point_bounds.max_first_point - simple_point_bounds.lower_bound
        dist_from_ub = simple_point_bounds.upper_bound - simple_point_bounds.min_last_point
        
        assert(dist_from_lb >= 0 and dist_from_lb == dist_from_ub)
        assert simple_point_bounds.max_first_point + abs_min_dist <= simple_point_bounds.upper_bound
        assert simple_point_bounds.lower_bound + abs_min_dist <= simple_point_bounds.min_last_point
        # print(f"Bounds 4: \n {simple_point_bounds!r}\n")
        
        point_min_width = simple_point_bounds.min_separation * (simple_point_bounds.max_points - 1)
        true_min_width = simple_point_bounds.fixed_width
        assert point_min_width >= true_min_width
        
        for constr, body_value, infeasible in find_infeasible_constraints(simple_point_bounds.model):
            raise ValueError(f"An infeasible constraint found: {str(constr)}: {body_value}. Type {infeasible}")
        
class TestBijection:
    def test_init(self, half_life_bounds):
        forward_func, inverse_func, bounds, y_min, y_max = half_life_bounds
        print(f"y_min = {y_min}, y_max = {y_max}")
        assert bounds.lower_bound == 0.0
        bijection = real_bijection.RealBijection(
            forward_func,
            bounds,
            inverse_func,
            compute_y_bounds=True
        )
        rtol = 1e-7 if bounds.dtype == np.float64 else 1e-5
        atol = 1e-8 if bounds.dtype == np.float64 else 1e-6
        calc_y_min, calc_y_max = bijection.y_bounds 
        assert calc_y_max > calc_y_min
        assert np.isclose(calc_y_min, y_min, rtol, atol), f"foward_func of x_max: {forward_func(bounds.upper_bound)}, calcated y_min {y_min}"
        assert np.isclose(calc_y_max, y_max, rtol, atol), f"foward_func of 0.0 {0.0}, calculated y_max {y_max}"
        assert bijection.direction is True #decreasing
        
        
        
        