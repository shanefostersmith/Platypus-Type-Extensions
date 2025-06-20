
import numpy as np
import pyomo.environ as pyo
from ._bounds_tools import (
    CascadePriority, BoundsViewMixin, BoundsState,
    _cascade_from_global, _cascade_from_points, _cascade_from_separation)
from pyomo.environ import value
from numbers import Number
from math import ceil
from typing import Literal, Optional, Union

#FUTURE WORK:
    # Defines bounds as Params of a pyomo model, and defines the constraints of bounds.
    # Could use model to tighten / find ideal bounds over time
    # Could add distribution functions as part of the model (instead of separately)
    
class PointBounds(BoundsViewMixin):
    """
    Define the bounds of a real number line, and bounds of *evenly-spaced* points on that line.
    
    These bounds may include:
    - **A lower bound and an upper bound** 
        - (the minimum value of the first point, and the maximum value of the last point)
    - **An upper bound of the first point** 
        - (greater than or equal to the lower bound)
    - **A lower bound of the last point**
        - (less than or equal to the upper bound)
    - **A minimum and maximum number of points**
        - (where the minimum is at least 2)
    - **The minimum and maximum distance between adjacent points**
        - (where the minimum is greater than 0)
    - **A fixed distance between the first and last point**
        - (less than or equal to than the span of the lower and upper bound)
    
    Other than `max_points` and `fixed_width`, all bounds are set at initialization.
    
    Automatic Bound Determination and Conflict Resolution
    ----
    When defining most or all of the bounds, conflicts may emerge. 
    Checking for these conflicts during every evolution and/or mutation is far less efficient than dealing with them beforehand.
    
    **To ensure consistent behavior across the classes and methods that use these bounds, this class will automatically resolve any conflicts.
    Bounds may also be set automatically if other attributes are avaliable.
        For example:**
    
        - Say the lower bound is `x0`, the upper bound is `xn`, and no fixed width has been set.
        - The upper bound of the first point is set to `xi` and the lower bound of the last point is set to `xj`
        - This gives you first point bounds `(x0, xi)` and last point bounds `(xj, xn)`. 
        Assuming `xi` < `xj` (note, this is not required in practice), then a minimum and maximum width for any set of points is implicity defined as:
            - `min_width = xj - xi`
            - `max_width =  xn - x0`
        - If a minimum number of points has been set, then the absolute maximum distance between two adjacent points is:
            - `max_separation = max_width / (min_points - 1)`
        - If a maximum number of points has been set, then the absolute minimum distance between two adjacent points is:
            - `min_separation = min_width / (max_points - 1)`
        - If the attributes for the minimum and maximum distance between adjacent points have not been set, or they fall outside these limits, they will be set/reset automatically.
        
    **Fixed Width Example**
        - The lower bound is set to `x0` and the upper bound is set to `xn`.
            Assuming `fixed_width` <= `xn - x0`, the upper bound of the first point `xi` and the lower bound of the last point `xj` are set to:
            - `xi = xn - fixed_width` 
            - `xj = x0 + fixed_width`
        - A fixed width implies that the first point bounds `(x0, xi)` and last point bounds `(xj, xn)` span the same distance.
        - A set of points can then "slide" within the first bound points / the last point bounds, but the width of those sets of points never changes.
            - If `x0 == xi` (implying `xj == xn`), then the first point will *always* be the lower bound, and the last point will *always* be the upper bound
    
    See the *Hierarchy* below for a reference on the "order of operations" for conflict resolutions and automatic setting of bounds.
    
    The Hierarchy:
    ---
    - **The lower bound and the upper bound (the full bounds)**
        - These bounds will never change automatically
        - If changed by the user, all other existing bounds will be checked and potentially changed to ensure compatibility.
    - **A fixed distance between the first and last point** 
        - This value is never set automatically.
        - Once set, this value will only change if the span of the full bounds becomes smaller than this width         
    - **The upper bound of the first point and the lower bound of the last point**
        - Are set automatically with the setting of a fixed width. If a fixed width is set, these bounds cannot be set until the fixed width reset to None.
        - These bounds will change if they become invalid due to setting full bound values, or due to setting a fixed width
        - If these bounds are not manually set, then they will be set automatically given the separation and cardinality bounds
    - **Bounds on number of points (cardinality) and bounds on distance between adjacent points (separation)**
        - Generally, setting cardinality bounds changes the separation bounds, and setting separation bounds changes the cardinality bounds (if applicable)
        - If the minimum and maximum separation bounds have not been set manually, then they will always be set to the minimum / maximum theoretical value.
        - *Adjustments only ensure that **at least one** combination of cardinality and separation values fall within the current width bounds*:
            - If width bounds are known, then it is required that:
                -  `min_separation` * `(max_points - 1)` >= `min_width` 
                -  `max_separation` * `(min_points - 1)` <= `max_width`
                -  `min_separation` <= `max_separation` and `min_points` <= `max_points`
                -  (Automatic adjustments are made to separation bounds and/or cardinality bounds so that these conditions are met)
            - **However, it is not required that**:
                - `min_separation` * `(min_points - 1)` >= `min_width`
                - `max_separation` * `(max_points - 1)` <= `max_width` 
                              
    Note on Precision
    ---
    - To be compatible with Numba components of this library, only numpy float32 and float64 types are supported
    - The absolute minimum separation between two numbers is given by: 
        -  `np.spacing(x, dtype = dtype)` where `x = max(abs(lower_bound), abs(upper_bound))`
        - This is to ensure all numbers between the lower and upper bound are representable with an even-spacing
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        inclusive_lower = True,
        inclusive_upper = True,
        minimum_points: int  = 2,
        maximum_points: Optional[int]  = None,
        precision: Literal['single', 'double'] = 'single'
    ):
        """
        Args:
            lower_bound (float): The minimum value of the first point
            upper_bound (float): The maximum value of the last point
            inclusive_lower (bool, optional): Defaults to True.
            inclusive_upper (bool, optional): Defaults to True.
            minimum_points (int, optional): Defaults to 2.
            maximum_points (int | None, optional): If None, an `np.inf` value is set. Defaults to None.
            precision (Literal[&#39;single&#39;, &#39;double&#39;], optional): The floating point precision type (float32 or float64). Defaults to 'single'.

        Raises:
            TypeError: If the input minimum points < 2
            TypeError: If the input maximum points < minimum points
        """              
        self.model = pyo.ConcreteModel()
        
        dtype = np.float32
        if precision == 'double':
            dtype = np.float64

        self._dtype = dtype
        lower_bound = dtype(lower_bound)
        upper_bound = dtype(upper_bound)
        eps = max(np.spacing(abs(lower_bound), dtype = dtype), np.spacing(abs(upper_bound), dtype = dtype))
        inclusive_lower_bound = lower_bound + eps
        inclusive_upper_bound = upper_bound - eps
        if inclusive_lower_bound >=  inclusive_upper_bound:
            raise TypeError(f"The inclusive lower bound {inclusive_lower_bound} >= the inclusive upper bound {inclusive_upper_bound}. Ensure precision is high enough for small widths")

        self.model.eff_lower = pyo.Param(initialize = inclusive_lower_bound if not inclusive_lower else lower_bound, mutable = True, within=pyo.Reals)
        self.model.eff_upper = pyo.Param(initialize = inclusive_upper_bound if not inclusive_upper else upper_bound, mutable = True, within=pyo.Reals)
        
        minimum_points = int(minimum_points)
        maximum_points = np.inf if maximum_points is None else int(maximum_points)
        if minimum_points < 2:
            raise TypeError(f"'minimum_points' must be greater than or equal to 2, got {minimum_points}")
        if  minimum_points > maximum_points:
            raise TypeError(f"'maximum_points' must be greater than or equal to minimum_points ({minimum_points}), got {maximum_points}")
        
        true_max_points = None
        try:
            true_max_points = int(np.floor(self.bound_width / eps + 1.0))
        except:
            true_max_points = np.inf
        
        min_separation = eps
        if not np.isinf(maximum_points):
            maximum_points =  min(true_max_points, maximum_points)
            minimum_points = min(minimum_points, maximum_points)
            # min_separation = (inclusive_upper_bound - inclusive_lower_bound) / dtype(maximum_points - 1)
        
        self.model.min_points = pyo.Param(initialize = minimum_points, mutable = True, within=pyo.PositiveIntegers)
        self.model.max_points = pyo.Param(initialize = maximum_points, mutable = True, within=pyo.PositiveReals)
        self.model.min_separation = pyo.Param(initialize = min_separation, mutable = True, within=pyo.PositiveReals)
        max_separation = (inclusive_upper_bound - inclusive_lower_bound) / dtype(minimum_points - 1)
        self.model.max_separation = pyo.Param(initialize = max_separation, mutable = True, within=pyo.PositiveReals)
        
        max_first = max(inclusive_lower_bound + eps, inclusive_upper_bound - (dtype(minimum_points - 1) * min_separation))
        min_last = min(inclusive_upper_bound  - eps, inclusive_lower_bound + (dtype(minimum_points - 1) * min_separation))
        self.model.max_first_point = pyo.Param(initialize = max_first, mutable = True, within=pyo.Reals)
        self.model.min_last_point = pyo.Param(initialize = min_last, mutable = True, within=pyo.Reals)
        
        self.model.fixed_width = pyo.Param(initialize = np.inf, mutable = True, within=pyo.PositiveReals)
        self.fixed_set = False
        self._manual_min_last = False
        self._manual_max_first = False
        self._manual_max_separation = False
        self._manual_min_separation = False
        self._set_all_constraints()
    
    def set_lower_bound(self, lower_bound: Number, inclusive = True, eps = 1e-8):
        """Set a new lower bound (the minimum value of the first point)
        
        Note, bounds are always stored and returned as *inclusive* bounds. 

        Args:
            lower_bound (Number): A lower bound. Must be a number < the set upper bound
            inclusive (bool, optional): If False, `eps` is added to the input lower bound. Defaults to True.
            eps (float, optional): A value to add to the lower bound if `inclusive` is False. Defaults to 1e-8.
        
        Raises:
            ValueError: If inclusive lower bound >= upper_bound
        """    
        
        lower_bound = self.dtype(lower_bound)  
        new_eps = self.get_separation_eps(lb_value = lower_bound)
        if not inclusive:
            lower_bound += new_eps
        if lower_bound >= self.upper_bound:
            raise ValueError("The input lower bound ({lower_bound}) >= the upper bound ({self.upper_bound})")
        max_val = self.upper_bound - self.dtype(2)*new_eps
        if lower_bound >= max_val:# Ensure no precision error
            new_eps = max(new_eps, np.spacing(max_val, dtype = self.dtype))
            lower_bound = self.upper_bound - self.dtype(2)*new_eps
        self.model.lower_bound = lower_bound
        
        self._cascade_from(CascadePriority.GLOBAL, 'min')
        
    def set_upper_bound(self, upper_bound: Number, inclusive = True):
        """Set a new upper bound (the maximum value of the last point)

        Note, bounds are always stored and returned as *inclusive* bounds.
        
        Args:
            upper_bound (Number): A lower bound. Must be a number > the set upper bound
            inclusive (bool, optional): If False, `eps` is subtracted from the input upper bound. Defaults to True.
            eps (float, optional): A value to subtract from the upper bound if `inclusive` is False. Defaults to 1e-8.
            
        Raises:
            ValueError: If inclusive upper bound <= lower_bound
        """   
        upper_bound = self.dtype(upper_bound)  
        new_eps = self.get_separation_eps(ub_value = upper_bound)
        if not inclusive:
             upper_bound -= new_eps 
        if upper_bound <= self.lower_bound:
            raise ValueError("The input lower bound ({lower_bound}) >= the upper bound ({self.upper_bound})")
        
        min_ub = self.lower_bound + self.dtype(2)*new_eps
        if upper_bound <= min_ub: # Ensure no precision error
            new_eps = max(new_eps, np.spacing(min_ub, dtype = self.dtype))
            upper_bound = self.lower_bound + self.dtype(2)*new_eps
            
        self.model.upper_bound = upper_bound
        self._cascade_from(CascadePriority.GLOBAL, 'max')   
    
    def set_first_point_upper_bound(self, max_first_point: Optional[Number]):    
        """
        Set the upper bound of the first point
        
        Will not be set if a fixed_width exists 
        
        If None is inputted, value is set to the max of:
            `lower_bound` and `upper_bound - (min_points - 1) * min_separation)`
        
        Raises:
            ValueError: If input > the upper bound
        """
        if self.fixed_set:
            return
        if max_first_point is not None and max_first_point > self.upper_bound:
            raise ValueError(f"The input max fisrt point ({max_first_point}) > the upper bound ({self.upper_bound}")
        eps = self.get_separation_eps()
        if max_first_point is None:
            self._manual_max_first = False
            max_first_point = max(self.lower_bound, self.upper_bound - self.dtype(self.min_points - 1) * self.min_separation)
        else:
            self._manual_max_first = True
            
        self.model.max_first_point = max(self.lower_bound, min(self.upper_bound - eps, self.dtype(max_first_point)))
        self._cascade_from(CascadePriority.WIDTH)
    
    def set_last_point_lower_bound(self, min_last_point: Optional[Number]):
        """
        Set the lower bound of the last point (min value of the last point)
        
        Will not be set if a fixed_width exists.
        
        If None is inputted, value is set to the min of:
            `upper_bound` and `lower_bound + (min_points - 1) * min_separation)`
        
        Raises:
            ValueError: If input < the lower bound
        """
        if self.fixed_set:
            return
        if min_last_point is not None and min_last_point < self.lower_bound:
            raise ValueError(f"The input min last point ({min_last_point}) < the lower bound ({self.lower_bound}")
        
        eps = self.get_separation_eps()
        if min_last_point is None:
            self._manual_min_last = False
            min_last_point = min(self.upper_bound, self.lower_bound + self.dtype(self.min_points - 1) * self.min_separation)
        else:
            self._manual_min_last = True
            
        self.model.min_last_point = min(self.upper_bound, max(self.lower_bound + eps, self.dtype(min_last_point)))  
        self._cascade_from(CascadePriority.WIDTH)

    def set_min_separation(self, min_separation: Optional[Number]):    
        """
        Set the min separation (i.e. the minimum distance between adjacent points)
        
        If None is inputted, will default to the smallest normalized number defined by the 'dtype'
        or the theoretical minimum if a min_width has been defined
        
        Raises:
            ValueError: If input is > the bound width or a set fixed width
        """
        bound_width = self.bound_width
        max_width = min(value(self.model.fixed_width), bound_width)
        eps = self.get_separation_eps()
        
        if min_separation is None: # set to abs minimum
            self._manual_min_separation = False
            if np.isinf(self.max_points):
                min_separation = eps
            elif self.fixed_width is not None:
                min_separation = min(eps, self.fixed_width / self.dtype(self.max_points - 1))
            else:
                min_width = 0
                strict_min_val = 0
                if self._manual_max_first:
                    min_width =  max(0, self.upper_bound - self._manual_max_first)
                if self._manual_min_last:
                    strict_min_val = self.min_last_point - self.max_first_point
                    min_width = max(max_width, self.min_last_point - self.lower_bound)
                    
                min_separation = eps if min_width <= eps else max(eps, min_width / self.dtype(self.min_points - 1))
                if strict_min_val > eps:
                    try:
                        min_separation = max(min_separation, strict_min_val / self.dtype(self.max_points - 1))
                    except: # overflow when max_points large
                        pass
                    
            min_separation = max(min_separation, eps)
            self.model.min_separation = self.dtype(min_separation)
            self.model.max_separation = max(min_separation, self.max_separation)
            self._cascade_from(CascadePriority.SEPARATION, 'min')
            return
                    
        elif min_separation > max_width:
            raise ValueError(f"Input min_separation {min_separation} > max_width {max_width}")
        elif min_separation <= 0:
            raise ValueError(f"Input min_separation <= 0")
        else:
            self._manual_min_separation = True
        
        min_separation = max(min_separation, eps)
        self.model.min_separation = self.dtype(min_separation)
        self.model.max_separation = max(min_separation, self.max_separation)
        self._cascade_from(CascadePriority.SEPARATION, 'min')
        
    def set_max_separation(self, max_separation: Optional[Number]):
        """
        Set the max separation (i.e. the maximum distance between adjacent points)
        
        If None is inputted, will default to theoretical maximum `max_width / (min_points - 1)`
        """
        bound_width = self.bound_width
        max_width = min(value(self.model.fixed_width), bound_width)
        eps = self.get_separation_eps()
        ub = max(eps, max_width / self.dtype(self.min_points - 1))
        
        
        if max_separation is None:
            max_separation = ub
            self._manual_max_separation = False
        elif max_separation <= 0:
            raise ValueError(f"Input max_separation <= 0")
        else:
            self._manual_max_separation = True
            max_separation = max(eps, min(self.dtype(max_separation), ub))
            
        self.model.max_separation = max_separation
        self.model.min_separation = min(self.min_separation, max_separation)
        self._cascade_from(CascadePriority.SEPARATION, 'max')
    
    def set_min_points(self, min_points: Optional[Number]): # Add option to change first last point boounds
        """
        Set the minimum number of points (inclusive)
            Will raise an error if min_points < 2
        
        If None is inputted, minimum points will default to 2
        """
        if min_points is None:
            min_points = 2
        min_points = int(min_points)
        if min_points < 2:
            raise ValueError(f"Min points must be > 1, got {min_points}")
        
        eps = self.get_separation_eps()
        try:
            min_points = min(min_points, max(1, int(self.bound_width / eps)) + 1)
        except: # overflow
            pass
        
        if not self._manual_max_separation: # set to max
            max_width = self.fixed_width or self.bound_width
            self.model.max_separation = max(eps, max_width / self.dtype(min_points - 1))
     
        self.model.min_points = min_points
        self.model.max_points = max(min_points, self.max_points)
        self._cascade_from(CascadePriority.POINTS, 'min')
    
    def set_max_points(self, max_points: Optional[Number]): # Add options to change first last point boounds
        """
        Set the maximum number of points (inclusive)
            Will raise an error if max_points < 2
        
        If None is inputted, max_points is set to a numpy inf number
        """
        if max_points is None:
            self.model.max_points = self.dtype(np.inf)
            return
        max_points = int(max_points)
        if max_points < 2:
            raise ValueError(f"Max points must be > 1, got {max_points}")
        
        eps = self.get_separation_eps()
        try:
            max_points = min(max_points, max(1, int(self.bound_width / eps)) + 1)
        except:
            pass
        
        self.model.max_points = max_points
        self.model.min_points = min(self.min_points, max_points)
        self._cascade_from(CascadePriority.POINTS, 'max')
        
    def set_fixed_width(self, width: Optional[Number]):
        """Set a fixed width (inclusive) 
        
        Automatically sets the "max_first_point" and "min_last_point".
        
        Will raise an error if 'width' is greater than the bound width
        """
        if width is None:
            self.fixed_set = False
            self.model.fixed_width = self.dtype(np.inf)
            return
        
        width = self.dtype(width)
        if width <= 0:
            raise ValueError(f"The input fixed_width {width} <= 0")
        lb = self.lower_bound
        ub = self.upper_bound
        bound_width = ub - lb
        
        if width > bound_width:
            raise ValueError(f"The input fixed_width {width} > the full bound width ({bound_width})")
        
        eps = self.get_separation_eps()
        abs_min = self.dtype(2)*eps
        if width <= abs_min:
            width = abs_min
            self.model.min_points = 2
            self.model.max_points = 2
            self.model.min_separation = eps
            self.model.max_separation = eps
            
        width_diff = bound_width - width
        if width_diff < abs_min:
            self.model.max_first_point = lb
            self.model.min_last_point = ub
            self.model.fixed_width = bound_width
        else:
            
            self.model.min_last_point = ub - width_diff
            self.model.max_first_point = lb + width_diff
            self.model.fixed_width = width
        
        
        self.fixed_set = True
        self._cascade_from(CascadePriority.WIDTH)
            
    def create_bounds_state(self) -> BoundsState:
        return BoundsState(
            _lower_bound     = self.lower_bound,
            _upper_bound     = self.upper_bound,
            _fixed_width     = self.fixed_width,
            _max_first_point = self.max_first_point,
            _min_last_point  = self.min_last_point,
            _min_separation  = self.min_separation,
            _max_separation  = self.max_separation,
            _min_points      = self.min_points,
            _max_points      = self.max_points,
            _dtype           = self.dtype
        )
        
    def _set_all_constraints(self):
        """Call at init only"""
        self.model.max_width_constr = pyo.Constraint(rule = _max_width_rule)
        self.model.min_width_constr = pyo.Constraint(rule = _min_width_rule)
        self.model.bound_constr = pyo.Constraint(rule = _bound_width_rule)
        
        self.model.min_point_constr = pyo.Constraint(expr = self.model.min_points >= 2)
        self.model.point_constr = pyo.Constraint(expr = self.model.min_points <= self.model.max_points)
        
        self.model.min_separation_constr = pyo.Constraint(rule = _min_separation_rule)
        self.model.separation_constr = pyo.Constraint(expr = self.model.min_separation <= self.model.max_separation)
        
        self.model.max_first_lb_constr = pyo.Constraint(expr = self.model.max_first_point >= self.model.eff_lower)
        self.model.max_first_ub_constr = pyo.Constraint(rule = _max_first_ub_rule)
        
        self.model.min_last_lb_constr  = pyo.Constraint(rule = _min_last_lb_rule)
        self.model.min_last_ub_constr = pyo.Constraint(expr = self.model.min_last_point <= self.model.eff_upper)
    
    def _cascade_from(self, level: CascadePriority, from_bound: Optional[Literal['max', 'min']] = None):
        bound_state = self.create_bounds_state()
        if level <= CascadePriority.GLOBAL:
            _cascade_from_global(
                bound_state, 
                from_min = False if not from_bound else from_bound == 'min',
                adjustable_max_first = not self._manual_max_first,
                adjustable_min_last = not self._manual_min_last,
                adjustable_min_separation = not self._manual_min_separation,
                adjustable_max_separation = not self._manual_max_separation)
        elif level <= CascadePriority.WIDTH:
            _cascade_from_points(
                bound_state, 
                from_max_points = None,
                adjustable_max_first = not self._manual_max_first,
                adjustable_min_last = not self._manual_min_last,
                adjustable_min_separation = not self._manual_min_separation,
                adjustable_max_separation = not self._manual_max_separation)
        elif level <= CascadePriority.POINTS:
            _cascade_from_points(
                bound_state, 
                from_max_points = None if not from_bound else from_bound == 'max',
                adjustable_max_first = not self._manual_max_first,
                adjustable_min_last = not self._manual_min_last,
                adjustable_min_separation = not self._manual_min_separation,
                adjustable_max_separation = not self._manual_max_separation)
        elif level <= CascadePriority.SEPARATION:
            _cascade_from_separation(
                bound_state, 
                from_max_sep = False if not from_bound else from_bound == 'max',
                adjustable_max_first = not self._manual_max_first,
                adjustable_min_last = not self._manual_min_last,
                adjustable_min_separation = not self._manual_min_separation,
                adjustable_max_separation = not self._manual_max_separation)
        self._apply_state(level, bound_state)
    
    def _apply_state(self, level: CascadePriority, bound_state: BoundsState):
        self.model.min_points = bound_state.min_points
        self.model.max_points = bound_state.max_points
        self.model.min_separation = bound_state.min_separation 
        self.model.max_separation = bound_state.max_separation 
        self.model.max_first_point = bound_state.max_first_point
        self.model.min_last_point = bound_state.min_last_point
        if level == CascadePriority.GLOBAL and self.fixed_set:
            self.model.fixed_width = bound_state.fixed_width
            
    @property
    def min_points(self) -> int:
        return value(self.model.min_points)

    @property
    def max_points(self) -> Union[int, np.floating]:
        return value(self.model.max_points)

    @property
    def lower_bound(self) -> np.floating:
        return value(self.model.eff_lower)

    @property
    def upper_bound(self) -> np.floating:
        return value(self.model.eff_upper)

    @property
    def fixed_width(self) -> Optional[np.floating]:
        return value(self.model.fixed_width) if self.fixed_set else None
    
    @property
    def min_last_point(self) -> np.floating:
        return value(self.model.min_last_point)
    
    @property
    def max_first_point(self) -> np.floating:
        return value(self.model.max_first_point)
    
    @property
    def min_separation(self) -> np.floating:
        return value(self.model.min_separation)
    
    @property
    def max_separation(self) -> np.floating:
        return value(self.model.max_separation)
    
    @property
    def dtype(self) -> type:
        return self._dtype
    
    def __repr__(self):
        return (
            f"First Point Bounds: ({self.lower_bound}, {self.max_first_point}), "
            f"Last Point Bounds: ({self.min_last_point}, {self.upper_bound}), "
            f"Cardinality Bounds: ({self.min_points}, {self.max_points}), "
            f"Separation Bounds: ({self.min_separation}, {self.max_separation}), "
            f"Fixed Width: {'None' if not self.fixed_set else f'{self.fixed_width}'}")
        
def _model_to_value(model):
    """
    Returns
        lower_bound value, upper_bound_value, dtype, eps
    """
    lb = value(model.eff_lower)
    ub = value(model.eff_upper)
    dtype = ub.dtype.type
    eps = max(np.spacing(abs(lb), dtype = dtype), np.spacing(abs(ub), dtype = dtype))
    return lb, ub, dtype, eps
        
def _max_width_rule(model):
    max_width = min(value(model.fixed_width), value(model.eff_upper) - value(model.eff_lower))
    return (model.min_points-1) * model.max_separation <= np.nextafter(max_width, np.inf)

def _min_width_rule(model):
    min_width = value(model.min_last_point) - value(model.max_first_point)
    return (model.max_points-1) * model.min_separation >= np.nextafter(min_width, -np.inf)

def _min_last_lb_rule(model):
    _, _, dtype, eps = _model_to_value(model)
    abs_min_dist = dtype(value(model.min_points) - 1) * eps
    return model.eff_lower <= value(model.max_first_point) - abs_min_dist

def _max_first_ub_rule(model):
    _, _, dtype, eps = _model_to_value(model)
    abs_min_dist = dtype(value(model.min_points) - 1) * eps
    return model.eff_upper >= value(model.max_first_point) + abs_min_dist

def _bound_width_rule(model):
    _, _, _, eps = _model_to_value(model)
    return model.eff_upper - model.eff_lower >= eps

def _min_separation_rule(model):
    _, _, dtype, eps = _model_to_value(model)
    return model.min_separation >= eps
    
