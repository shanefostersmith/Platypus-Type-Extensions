
import numpy as np
from dataclasses import dataclass
from math import ceil, floor
from enum import IntEnum
from typing import Union, Optional

class CascadePriority(IntEnum):
    GLOBAL = 0
    WIDTH = 1
    POINTS = 2
    SEPARATION = 3
    
class BoundsViewMixin:
    """Defines shared properties / getter methods for PointBounds and BoundsState"""
    @property
    def min_points(self) -> int:
        raise NotImplementedError

    @property
    def max_points(self) -> Union[int, np.floating]:
        raise NotImplementedError

    @property
    def lower_bound(self) -> np.floating:
        raise NotImplementedError

    @property
    def upper_bound(self) -> np.floating:
        raise NotImplementedError

    @property
    def fixed_width(self) -> Optional[np.floating]:
        raise NotImplementedError
    
    @property
    def min_last_point(self) -> np.floating:
        raise NotImplementedError
    
    @property
    def max_first_point(self) -> np.floating:
        raise NotImplementedError
    
    @property
    def min_separation(self) -> np.floating:
        raise NotImplementedError
    
    @property
    def max_separation(self) -> np.floating:
        raise NotImplementedError
    
    @property
    def dtype(self) -> type:
        raise NotImplementedError
    
    @property
    def bound_width(self) -> np.floating:
        """The difference between the upper and lower bound"""
        return self.upper_bound - self.lower_bound
    
    @property
    def true_min_width(self):
        """Minimum width accounting for all of the separation, cardinality and first/last point bounds"""
        if self.fixed_width:
            return self.fixed_width
        
        min_width_by_separation = self.dtype(self.min_points- 1) * self.min_separation
        if self.max_first_point < self.min_last_point:
            return max(self.max_first_point - self.min_last_point, min_width_by_separation)
        return min_width_by_separation
    
    @property
    def true_max_width(self):
        """Maximum width account for all of the separation, cardinality and lower/upper bounds"""
        if self.fixed_width:
            return self.fixed_width
        
        w = self.bound_width
        if not np.isinf(self.max_points):
            max_width_by_separation = self.dtype(self.max_points - 1) * self.max_separation
            return min(max_width_by_separation, w)
        return w
    
    @property
    def first_point_bounds(self) -> tuple:
        """inclusive bounds of the first point: `(minimum first point, maximum first point)`"""
        return self.lower_bound, self.max_first_point
    
    @property
    def last_point_bounds(self) -> tuple:
        """inclusive bounds of the last point: `(minimum last point, maximum last point)`"""
        return self.min_last_point, self.upper_bound
    
    def get_separation_eps(self, lb_value = None, ub_value  = None):
        """Get the true smallest separation within current or new lower and upper bound values
         """
        lb = self.dtype(lb_value) if lb_value is not None else self.lower_bound
        ub = self.dtype(ub_value) if ub_value is not None else self.upper_bound
        return max(np.spacing(abs(lb), dtype = self.dtype), np.spacing(abs(ub), dtype = self.dtype))
    
    def get_conditional_separation_bounds(self, num_points: int) -> tuple[np.floating, np.floating]:
        """Find the separation bounds given a fixed number of points"""
        assert num_points > 1, "input must be greater than 1"
        denom = self.dtype(num_points - 1)
        if self.fixed_width is not None:
            out = self.fixed_width / denom 
            return out, out

        out_min_separation = None
        if self.max_first_point >= self.min_last_point:
            out_min_separation = self.min_separation
        else:
            out_min_separation = (self.min_last_point - self.max_first_point) / denom

        max_width = self.true_max_width
        out_max_separation = max(out_min_separation, max_width / denom)
        return out_min_separation, out_max_separation
    
    def get_conditional_cardinality_bounds(self, separation: float):
        """Get the bounds on the number of points given a fixed separation value.
        
        Note, if the difference between the min_width and max_width is small (or 0 when fixed_width a is set),
        then there might not be a number of points where:
            `(point - 1) * separation` is in the range `(min_width, max_width)`.
            
        This function finds the closest number of points"""
        
        denom = self.dtype(separation)
        assert denom > 0, "input must be > 0"
        if self.fixed_width is not None:
            out_points = int(self.fixed_width / denom + 1)
            return out_points, out_points
        
        max_width = self.true_max_width
        max_points = int(max_width / denom + 1)
        if not np.isinf(max_points):
            max_points = min(self.max_points, max_points)
        min_width = self.true_min_width
        min_points = max(self.min_points, min(self.max_points, int(ceil(min_width / denom + 1.0))))
        max_points = max(max_points, min_points)
        return min_points, max_points
    
    def get_conditional_cardinality_with_width(self, width: float) -> tuple[int, int]:
        """Get the bounds on the number of points given a fixed width (using separation bounds)"""
        
        width = self.dtype(width)
        assert width > 0, "input must be > 0"
        
        max_points = None
        try:
            max_points = max(self.min_points, min(self.max_points, max(1, int(width  / self.min_separation)) + 1))
        except:
            max_points = self.max_points

        min_points = min(max_points, max(self.min_points, ceil(width / self.max_separation) + 1))
        return min_points, max_points
    
    def get_conditional_separation_with_width(self, width: float) -> tuple[np.floating, np.floating]:
        """Get separation bounds given a fixed width (using cardinality bounds)"""
        num = self.dtype(width)
        # assert num > 0, "input must be > 0"
        
        min_separation = None
        if np.isinf(self.max_points):
            min_separation = self.min_separation
        else:
            min_separation = width / self.dtype(self.max_points - 1)
        max_separation = max(min_separation, num / self.dtype(self.min_points - 1))
        return min_separation, max_separation


@dataclass
class BoundsState(BoundsViewMixin):
    """A compact view of a PointBound class. Does not contain setter methods, but has the same properties and getter methods"""
    
    __slots__ = (
        "_lower_bound", "_upper_bound", "_fixed_width",
        "_max_first_point", "_min_last_point",
        "_min_separation", "_max_separation",
        "_min_points", "_max_points", "_dtype"
    )
    _lower_bound: np.floating
    _upper_bound: np.floating
    _fixed_width: Optional[np.floating] 
    _max_first_point: np.floating
    _min_last_point: np.floating
    _min_separation: np.floating
    _max_separation: np.floating
    _min_points: int
    _max_points: Union[int,np.floating]
    _dtype: type
    
    @property
    def min_points(self) -> int:
        return self._min_points

    @property
    def max_points(self) -> Union[int,np.floating]:
        return self._max_points

    @property
    def lower_bound(self) -> np.floating:
        return self._lower_bound

    @property
    def upper_bound(self) -> np.floating:
        return self._upper_bound

    @property
    def fixed_width(self) -> Optional[np.floating]:
        return self._fixed_width
    
    @property
    def min_last_point(self) -> np.floating:
        return self._min_last_point
    
    @property
    def max_first_point(self) -> np.floating:
        return self._max_first_point
    
    @property
    def min_separation(self) -> np.floating:
        return self._min_separation
    
    @property
    def max_separation(self) -> np.floating:
        return self._max_separation
    
    @property
    def dtype(self) -> type:
        return self._dtype

'''
Applies cascading logic outside PointBounds class. Better for testing.
'''

def _cascade_from_global(bounds: BoundsState, from_min = None,
                        adjustable_max_first = False, 
                        adjustable_min_last = False,
                        adjustable_min_separation = True,
                        adjustable_max_separation = True):
    """
    Called when changing lower or upper bound.
    Assumes lower_bound < upper_bound"""

    eps = bounds.get_separation_eps()
    
    # Smallest width edge case
    abs_min_dist = bounds.dtype(2) * eps
    if bounds.lower_bound + abs_min_dist >= bounds.upper_bound:
        if from_min or abs(bounds.upper_bound) >= abs(bounds.lower_bound):
            bounds._upper_bound = bounds._lower_bound + abs_min_dist
        else:
            bounds._lower_bound = bounds._upper_bound - abs_min_dist
        eps = bounds.get_separation_eps()
        bounds._min_points = 2
        bounds._max_points = 2
        bounds._max_separation = eps
        bounds._min_separation = eps
        bounds._max_first_point = bounds._lower_bound if bounds._max_first_point <= bounds.lower_bound or bounds.fixed_width else bounds.lower_bound + eps
        bounds._min_last_point = bounds._upper_bound if bounds._min_last_point >= bounds.upper_bound or bounds.fixed_width else bounds.lower_bound + eps
        if bounds._fixed_width:
            bounds._fixed_width = abs_min_dist
            bounds._max_first_point = bounds._lower_bound
            bounds._min_last_point = bounds._upper_bound
        else:
            bounds._max_first_point = bounds._lower_bound if bounds._max_first_point <= bounds.lower_bound else bounds.lower_bound + eps
            bounds._min_last_point = bounds._upper_bound if bounds._min_last_point >= bounds.upper_bound  else bounds.lower_bound + eps
        return
        
    if bounds._fixed_width:
        bound_width = bounds.bound_width
        width_diff = bound_width - bounds._fixed_width
        if width_diff < abs_min_dist:
            bounds._fixed_width = bound_width
            bounds._max_first_point = bounds._lower_bound
            bounds._min_last_point = bounds._upper_bound
        else:
            bounds._fixed_width = max(abs_min_dist, min(bounds._fixed_width, bound_width))
            if bounds._fixed_width < abs_min_dist:
                bounds._fixed_width = eps
                bounds._max_first_point = bounds._upper_bound - eps
                bounds._min_last_point = bounds._lower_bound + eps
            else:
                bounds._max_first_point = bounds._lower_bound + width_diff
                bounds._min_last_point = bounds._upper_bound - width_diff
    else:
        # First point upper bound adjustment
        if bounds._lower_bound + eps >= bounds._max_first_point:
            bounds._max_first_point = bounds._lower_bound
        elif bounds._max_first_point > bounds._upper_bound - eps:
            bounds._max_first_point = bounds._upper_bound - eps

        # Last point lower bound adjustment
        if bounds._min_last_point + eps >= bounds._upper_bound:
            bounds._min_last_point = bounds._upper_bound
        elif not bounds._min_last_point > bounds._lower_bound + eps:
            bounds._min_last_point  = bounds._lower_bound + eps

    _cascade_from_points(
        bounds, None, 
        adjustable_max_first, 
        adjustable_min_last,
        adjustable_min_separation,
        adjustable_max_separation
    )  

def _cascade_from_points(
    bounds: BoundsState, from_max_points = None, 
    adjustable_max_first = False, 
    adjustable_min_last = False,
    adjustable_min_separation = True,
    adjustable_max_separation = True):
    """
    Assumes all non-cardinality / non-separation bounds are valid. 
    
    Assumes 1 < min_points <= max_points

    Adjusts separation and points
    """
    eps = bounds.get_separation_eps()
    max_width = bounds._fixed_width or bounds._upper_bound - bounds._lower_bound
    min_width = bounds._fixed_width or max(eps, bounds._min_last_point - bounds._max_first_point)
    max_only = _check_first_last_bounds(
        bounds, max_width, eps,
        adjustable_max_first, 
        adjustable_min_last,
        adjustable_max_separation
    )
    
    if max_only or max_width <= bounds.dtype(2) * eps: # adjust max_separation / max_points only
        _min_points_tool(bounds, max_width, eps)
        _min_sep_tool(bounds, min_width)
    elif from_max_points:
        _max_points_tool(bounds, min_width, eps, adjustable_min_separation)
        _min_points_tool(bounds, max_width, eps)
    else:
        _min_points_tool(bounds, max_width, eps)
        _max_points_tool(bounds, max_width, eps, adjustable_min_separation)
    
    # Minimize min_separation if adjustable
    if adjustable_min_separation and not (
        bounds._fixed_width or not bounds._min_separation > eps):
        
        min_width = max(eps, bounds._min_last_point - bounds._max_first_point)
        if not min_width > eps or np.isinf(bounds._max_points):
            bounds._min_separation = eps
        else:
            bounds._min_separation = max(eps, min_width / bounds.dtype(bounds._max_points - 1))
        
        new_min_w_points = bounds._dtype(bounds._min_points - 1) * bounds._min_separation
        if adjustable_max_first:
            bounds._max_first_point = max(bounds.lower_bound, bounds._upper_bound - new_min_w_points)
        if adjustable_min_last:
            bounds._min_last_point = min(bounds._upper_bound, bounds._lower_bound + new_min_w_points)
    

def _cascade_from_separation(
    bounds: BoundsState, from_max_sep: bool,
    adjustable_max_first = False, 
    adjustable_min_last = False,
    adjustable_min_separation = True,
    adjustable_max_separation = True):
    """ Called from adjusting separation, can find max_points
    
    Assumes all non-cardinality / non-separation bounds are valid
    
    Adjusts separation and points
    """
    eps = bounds.get_separation_eps() 
    max_width = bounds._fixed_width or bounds._upper_bound - bounds._lower_bound
    min_width = bounds._fixed_width or max(eps, bounds._min_last_point - bounds._max_first_point)
    max_only = _check_first_last_bounds(
        bounds, max_width, eps, 
        adjustable_max_first, 
        adjustable_min_last, 
        from_min_sep = not from_max_sep)
    
    if max_only or max_width <= bounds.dtype(2) * eps: # adjust max_separation / max_points only
        _min_sep_tool(bounds, min_width)
        _min_points_tool(bounds, max_width, eps)
    elif from_max_sep:
        _max_sep_tool(bounds, max_width)
        _min_sep_tool(bounds, min_width)
    else:
        _min_sep_tool(bounds, min_width)
        _max_sep_tool(bounds, max_width)
    
    # If min/max width difference / large separations, cannot achieve width bounds evenly. Adjust separations as well
    if (bounds.dtype(bounds._min_points - 1) * bounds._max_separation > max_width or 
        (not np.isinf(bounds._max_points) and bounds.dtype(bounds._max_points - 1) * bounds._min_separation < min_width)
    ):
        _cascade_from_points(
        bounds, 
        None,
        adjustable_max_first, 
        adjustable_min_last,
        adjustable_min_separation,
        adjustable_max_separation)
    
    
def _max_sep_tool(bounds: BoundsState, max_width):
    """
    checks: `(min_points - 1) * bounds._max_separation <= max_width`
    
    Decrease min_points to fit max_width. Assumes max_separation <= max_width
    """
    if bounds._min_points == 2:
        return
    curr_dist = (bounds._min_points - 1) * bounds._max_separation
    if curr_dist > max_width:
        bounds._min_points = max(2, int(max_width / bounds._max_separation + 1.0))
        bounds._max_points = max(bounds._min_points, bounds._max_points)
        
def _min_sep_tool(bounds: BoundsState, min_width):
    """
    checks: `(max_points - 1) * min_separation >= min_width`
    
    Increase max_points to fit min_width
    
    Assumes min_separation >= separation eps and min_points <= max_points"""
    # print("min_sep_tool")
    if np.isinf(bounds._max_points):
        return
    curr_dist = (bounds._max_points - 1) * bounds._min_separation
    if curr_dist < min_width:
        dist_to_min = min_width - curr_dist
        min_addition_points = int(ceil(dist_to_min / bounds._min_separation)) + 1
        bounds._max_points += min_addition_points
        bounds._min_points = min(bounds._min_points, bounds._max_points)
    
        
def _min_points_tool(bounds: BoundsState, max_width, eps):
    """
    checks: `(min_points - 1) * bounds._max_separation <= max_width`
    
    Decrease max separation to fit max_width
    """
    # print("min_points_tool")
    curr_dist = bounds._max_separation * bounds.dtype(bounds._min_points - 1)
    if curr_dist > max_width:
        abs_max_separation = max_width / bounds.dtype(bounds._min_points - 1)
        if abs_max_separation <= eps:
            assert bounds._min_points > 2, "Error: width too small, cannot adjust max_separation or min_points"
            bounds._max_separation = eps
            bounds._min_separation = eps
            bounds._min_points = int(max_width / eps + 1.0)
        else:
            bounds._max_separation = abs_max_separation
            bounds._min_separation = max(eps, min(bounds._min_separation, bounds._max_separation))
            
            if not bounds._max_separation * (bounds._min_points - 1) <= max_width: # rounding/precision issue
                bounds._max_separation = np.nextafter(abs_max_separation, -np.inf, dtype = bounds._dtype)
                bounds._min_separation = max(eps, min(bounds._min_separation, bounds._max_separation))
    
def _max_points_tool(
    bounds: BoundsState, 
    min_width, eps,
    adjustable_min_separation = False):
    """
    checks: `(max_points - 1) * min_separation >= min_width`
    
    If max_point not inf, increase min separation to fit min_width.
    Assumes min_separation >= eps"""
    if np.isinf(bounds._max_points):
        return
    curr_dist = bounds.dtype(bounds._max_points - 1) * bounds._min_separation
    if curr_dist < min_width or adjustable_min_separation:
        # print("max_points tool")
        bounds._min_separation = max(eps, min_width / bounds.dtype(bounds._max_points - 1))
        bounds._max_separation = max(bounds._max_separation, bounds._min_separation)
        
        if not bounds._min_separation * bounds.dtype(bounds._max_points - 1) >= min_width: # rounding/precision issue
            bounds._min_separation = np.nextafter(bounds._min_separation, np.inf, dtype = bounds._dtype)
            bounds._max_separation = max(bounds._max_separation, bounds._min_separation)
                
def _check_first_last_bounds(bounds: BoundsState, 
                             max_width, eps,
                             adjustable_max_first = False, 
                             adjustable_min_last = False,
                             adjustable_max_separation = True,
                             from_min_sep = False) -> bool:
    """Checks if distances from max_first to ub and min_last to lb, adjusts min_points or min_separation if applicable
    
    If not fixed_width, ensure that `(min_points - 1) * min_separation <= min(ub - max_first, min_last - lb)`
    
    May alter max_first or min_last if adjustable.
    
    (Does not deal with min widths defined by min_last - max_first)
    
    Assumptions:
    - 
    
    (max_points - 1) * min_separation >= min_width
    (min_points - 1) * max_separation >= max_width
    Returns:
        -  boolean indicating if min_points / min_separation were adjusted and should only adjust max_points / max_separation"""
    
    # Check if min_points / min_separation are valid with max_first and min_last bounds
    bounds._max_separation = min(max_width, max(bounds._max_separation, eps))
    bounds._min_separation = max(eps, min(bounds._min_separation, bounds.max_separation))
    
    dist_from_lb = bounds._min_last_point - bounds._lower_bound
    dist_from_ub = bounds._upper_bound - bounds._max_first_point
    if bounds._fixed_width:
        return min(dist_from_lb, dist_from_ub) <= eps

    if dist_from_lb < eps:
        dist_from_lb = eps
        bounds._min_last_point = bounds._lower_bound + eps
        
    dist_from_ub = bounds._upper_bound - bounds._max_first_point
    if dist_from_ub < eps:
        dist_from_ub = eps
        bounds._max_first_point = bounds.upper_bound - eps
    
    curr_min_w_points = bounds._dtype(bounds._min_points - 1) * bounds._min_separation
    max_width = 0
    if not adjustable_max_first:
        max_width = dist_from_ub
    if not adjustable_min_last:
        max_width = min(max_width, dist_from_lb) if not adjustable_max_first else dist_from_lb
    
    if max_width == 0: # can adjust_either
        bounds._min_last_point = min(bounds._upper_bound, bounds._lower_bound + curr_min_w_points)
        bounds._max_first_point = max(bounds.lower_bound, bounds._upper_bound - curr_min_w_points)
        return False
    
    max_only = False
    if not max_width > eps:
        bounds._min_points = 2
        bounds._min_separation = eps
        max_only = True
    
    elif from_min_sep and not bounds._min_separation < max_width:
        bounds._min_points = 2
        bounds._min_separation = max_width
        max_only = True
        
    elif from_min_sep:
        bounds._min_points = int(max(1, max_width / bounds._min_separation)) + 1
        if bounds._dtype(bounds._min_points - 1) * bounds._min_separation > max_width:
            bounds._min_separation = max(eps, max_width / bounds._dtype(bounds._min_points - 1))
            max_only = True
    else:
        temp_separation = max_width / bounds._dtype(bounds._min_points - 1)
        bounds._min_separation = max(temp_separation, eps)
        if temp_separation < eps:
            bounds._min_points = int(max(1, max_width / bounds._min_separation)) + 1
            max_only = True
    
    new_min_w_points = bounds._dtype(bounds._min_points - 1) * bounds._min_separation
    if adjustable_max_first:
        bounds._max_first_point = max(bounds.lower_bound, bounds._upper_bound - new_min_w_points)
    if adjustable_min_last:
        bounds._min_last_point = min(bounds._upper_bound, bounds._lower_bound + new_min_w_points)
    if adjustable_max_separation:
        bounds._max_separation = bounds.bound_width / bounds._dtype(bounds._min_points - 1)
    
    bounds._max_separation = max(bounds._min_separation, bounds._max_separation)
    bounds._max_points = max(bounds._min_points, bounds._max_points)
    return max_only
    
    
            