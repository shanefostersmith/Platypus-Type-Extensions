
import numpy as np
from dataclasses import dataclass
from math import ceil
from enum import IntEnum

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
    def max_points(self) -> int | np.floating:
        raise NotImplementedError

    @property
    def lower_bound(self) -> np.floating:
        raise NotImplementedError

    @property
    def upper_bound(self) -> np.floating:
        raise NotImplementedError

    @property
    def fixed_width(self) -> np.floating | None:
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
    
    def get_true_min_width(self):
        """Get the minimum width accounting for all of the separation, cardinality and first/last point bounds"""
        if self.fixed_width:
            return self.fixed_width
        
        min_width_by_separation = self.dtype(self.min_points- 1) * self.min_separation
        if self.max_first_point < self.min_last_point:
            return max(self.max_first_point - self.min_last_point, min_width_by_separation)

        return min_width_by_separation
    
    def get_true_max_width(self):
        """Get the maximum width account for all of the separation, cardinality and lower/upper bounds"""
        if self.fixed_width:
            return self.fixed_width
        if not np.isinf(self.max_points):
            max_width_by_separation = self.dtype(self.max_points - 1) * self.max_separation
            return min(max_width_by_separation, self.get_bound_width())

        return self.get_bound_width()
    
    def get_first_point_bounds(self) -> tuple:
        """Get the inclusive bounds of the first point: `(minimum first point, maximum first point)"""
        return self.lower_bound, self.max_first_point
    
    def get_last_point_bounds(self) -> tuple:
        """Get the inclusive bounds of the last point: `(minimum last point, maximum last point)"""
        return self.min_last_point, self.upper_bound
    
    def get_full_bounds(self) -> tuple:
        """Get the *inclusive* lower and upper bounds"""
        return self.lower_bound, self.upper_bound
    
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
            out_min_separation = (self.max_first_point - self.min_last_point) / denom
        
        max_width = self.get_true_max_width()
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
        
        max_width = self.get_true_max_width()
        max_points = int(max_width / denom + 1)
        if not np.isinf(max_points):
            max_points = min(self.max_points, max_points)
        min_width = self.get_true_min_width()
        min_points = int(ceil(min_width / denom + 1))
        max_points = max(max_points, min_points)
        return min_points, max_points
    
    def get_conditional_cardinality_with_width(self, width: float) -> tuple[int, int]:
        """Get the bounds on the number of points given a fixed width (using separation bounds)"""
        
        num = self.dtype(width)
        assert num > 0, "input must be > 0"
        
        max_points = int(num / self.max_separation + 1)
        min_points = min(max_points, int(ceil(num / self.min_separation + 1)))
        return min_points, max_points
    
    def get_conditional_separation_with_width(self, width: float) -> tuple[np.floating, np.floating]:
        """Get separation bounds given a fixed width (using cardinality bounds)"""
        num = self.dtype(width)
        assert num > 0, "input must be > 0"
        
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
    _fixed_width: np.floating | None
    _max_first_point: np.floating
    _min_last_point: np.floating
    _min_separation: np.floating
    _max_separation: np.floating
    _min_points: int
    _max_points: int | np.floating
    _dtype: type
    
    @property
    def min_points(self) -> int:
        return self._min_points

    @property
    def max_points(self) -> int | np.floating:
        return self._max_points

    @property
    def lower_bound(self) -> np.floating:
        return self._lower_bound

    @property
    def upper_bound(self) -> np.floating:
        return self._upper_bound

    @property
    def fixed_width(self) -> np.floating | None:
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

def _cascade_from_global(bounds: BoundsState):
    """Called when changing lower or upper bound.
    
    Assumes lower_bound < upper_bound"""
    eps = np.finfo(bounds.dtype).tiny
    
    # Fix first/last point_bounds
    if bounds._fixed_width:
        if bounds._fixed_width > bounds._upper_bound - bounds._lower_bound:
            bounds._fixed_width = bounds._upper_bound - bounds._lower_bound
            bounds._max_first_point = bounds._lower_bound
            bounds._min_last_point = bounds._upper_bound
        else:
            bounds._max_first_point = bounds._upper_bound - bounds._fixed_width
            bounds._min_last_point = bounds._lower_bound + bounds._fixed_width
    else:
        if bounds._lower_bound > bounds._max_first_point:
            bounds._max_first_point = bounds._lower_bound
        else:
            bounds._max_first_point = bounds._dtype(min(
                bounds._max_first_point,
                bounds._upper_bound- (bounds._min_points - 1)*eps)
            )
        if bounds._upper_bound < bounds._min_last_point:
            bounds._min_last_point = bounds._upper_bound
        else:
            bounds._min_last_point = bounds._dtype(max(
                bounds._min_last_point,
                bounds._lower_bound + (bounds._min_points - 1) * eps)
            )
    _cascade_from_points(bounds)  
    
def _cascade_from_points(bounds: BoundsState, from_max_points = None):
    """Assumes all non-cardinality / non-separation bounds are valid. 
    Assumes 1 < min_points <= max_points
    
    Called from changing width (other than global) and setting new points,
        (internally sometimes setting separation)
    
    Adjusts separations, if applicable"""
    eps = np.finfo(bounds._dtype).tiny
    max_width = bounds._fixed_width or bounds._upper_bound - bounds._lower_bound
    min_width = bounds._fixed_width or max(eps, bounds._min_last_point - bounds._max_first_point)
    
    if from_max_points:
        _max_points_tool(bounds, min_width)
        _min_points_tool(bounds, max_width)
    else:
        _min_points_tool(bounds, max_width)
        _max_points_tool(bounds, min_width)
    
def _cascade_from_separation(bounds: BoundsState, from_max_sep: bool):
    """ Called from adjusting separation, can find max_points
    
    Assumes all valid widths
    
    Assumes 0 < min_separation <= max_separation <= full width / fixed width"""
    eps = np.finfo(bounds._dtype).tiny
    max_width = bounds._fixed_width or bounds._upper_bound - bounds._lower_bound
    min_width = bounds._fixed_width or max(eps, bounds._min_last_point - bounds._max_first_point)
    
    if from_max_sep:
        _max_sep_tool(bounds, min_width)
        _min_sep_tool(bounds, max_width)
    else:
        _min_sep_tool(bounds, max_width)
        _max_sep_tool(bounds, min_width)
    
    # If small width difference or large separations 
    # possible that cannot get within width bounds evenly
    if max_width - min_width <= bounds.max_separation: 
        _cascade_from_points(bounds)

def _min_sep_tool(bounds: BoundsState, max_width):
    points_ub = int(max_width / bounds._min_separation + 1.0)
    if not bounds._max_points or bounds._max_points > points_ub:
        bounds._max_points = int(max_width / bounds._min_separation + 1.0)
        bounds._min_points = min(bounds._min_points, bounds._max_points)

def _max_sep_tool(bounds: BoundsState, min_width):
    points_lb = int(ceil(min_width / bounds._max_separation + 1.0)) 
    if points_lb > bounds._min_points:
        bounds._min_points = points_lb
        bounds._max_points = max(bounds._max_points,  bounds._min_points)

def _min_points_tool(bounds: BoundsState, max_width):
    temp_ub = bounds._max_separation * (bounds._min_points - 1)
    if temp_ub > max_width:
        abs_max_separation = max_width / bounds.dtype(bounds._min_points - 1)
        bounds._max_separation = abs_max_separation
        bounds._min_separation = min(bounds._min_separation, bounds._max_separation)
        
def _max_points_tool(bounds: BoundsState, min_width):
    if not np.isinf(bounds._max_points):
        temp_lb = bounds._min_separation * (bounds._max_points - 1)
        if temp_lb < min_width:
            bounds._min_separation = min_width / bounds._dtype(bounds._max_points - 1)
            bounds._max_separation = max(bounds._min_separation, bounds._max_separation)
    else:
        bounds._min_separation = min(bounds._min_separation, bounds._max_separation)