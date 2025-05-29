import numpy as np
from warnings import warn
from typing import Union, Tuple
from numbers import Integral, Number
from math import ceil

'''
(Previous point bounds:
Too much dependency / hard to test
Useful features with infinite bounds and exclusivity.
Does not use pyo)
'''




class PointBounds:
    """ 
    """
    def __init__(self,
        lower_bound: Union[float, np.number] = None, 
        upper_bound: Union[float, np.number] = None, 
        inclusive_lower_bound: bool = True, 
        inclusive_upper_bound: bool = True,
        minimum_points: Union[int, np.integer, None] = None,
        maximum_points: Union[int, np.integer, None] = None,
        stricter_points: bool = True,
        dtype: type = np.float64,
        integer_dtype = np.uint32,
        show_warnings = False
    ):
        # Set numpy float type
        if dtype == float:
            dtype = np.float64
        elif not issubclass(dtype, np.number):
            raise ValueError(f"'Invalid 'dtype' provided {dtype}")
        elif issubclass(dtype, Integral):
            raise ValueError("Integer types are not currently supported")
        self.dtype = dtype
        self.eps = np.finfo(dtype).tiny
        
        # Set full bounds
        self._bounds = tuple([None if b is None else dtype(b) for b in (lower_bound, upper_bound)])
        if self.bound_is_finite():
            assert self._bounds[0] < self._bounds[1] , f"Lower bound ({self._bounds[0]}) is not less than upper bound ({self._bounds[1]})"
        
        assert isinstance(inclusive_lower_bound, (bool, np.bool_)), "Lower inclusivity not a boolean"
        assert isinstance(inclusive_upper_bound, (bool, np.bool_)), "Upper inclusivity not a boolean"
        self._inclusivity = (inclusive_lower_bound, inclusive_upper_bound)
        if self.bound_is_finite():
            assert self.get_calculable_lower_bound() < self.get_calculable_upper_bound(), f"With the current 'dtype' and inclusivity inputs, lower bound ({self._bounds[0]}) is not less than upper bound ({self._bounds[1]})"
        
        # Set numpy integer (for points)
        if integer_dtype == int:
            integer_dtype = np.uint64
        elif not issubclass(integer_dtype, np.number):
            raise ValueError(f"'Invalid 'dtype' provided {dtype}")
        elif not issubclass(integer_dtype, Integral):
            raise ValueError("Points must be an integer type")
        self.integer_dtype = integer_dtype
        
        # Set point boundsd
        self._point_bounds = tuple([None if b is None else integer_dtype(b) for b in (minimum_points, maximum_points)])
        if self._point_bounds[0] is not None:
            assert 2 <= self._point_bounds[0], f"The minimum number of points must be at least 2 (input was [{self._point_bounds[0]})"
            if self._point_bounds[1] is not None:
                assert self._point_bounds[0] <= self._point_bounds[1], f"The minimum number of points must be less than the maximum_points (input was [{self._point_bounds[0]},{self._point_bounds[1]}]"
        
        self._min_separation = None
        self._max_separation = None if not (self.bound_is_finite() and self._point_bounds[0]) else self.get_bound_width() / self.dtype(self._point_bounds[0])
   
        # Upper bound first point
        self._max_first_point = None
        self._max_first_inclusive = True
        
        # Lower bound last point
        self._min_last_point = None
        self._min_last_inclusive = True
        
        # Other
        self._fixed_width = None
        self._stricter_points = stricter_points
        self._show_warnings = show_warnings
         
    def set_full_bounds(self, 
        lower_bound: Number | None, upper_bound: Number | None, 
        inclusive_lower_bound = None, inclusive_upper_bound = None):
        """ `
        Set new lower and upper bounds
            (represent the minimum value of the first point and maximum value of the last point)
        
        Will raise an error if lower_bound >= upper_bound
        
        Bounds set to `None` represents open / infinite bounds. 
        
        If inclusive_lower_bound and/or inclusive_upper_bound are None, then the previous inclusivity is used
        
        **Note, both bounds will be reset**:   
            - For example, if the previous lower bound is not None, but the input lower bound is None, the lower bound will be set to None
            
            - If you want to reset one bound without reseting the other, provide one new bound and one previous bound.
        
        Any attributes that become invalid as a result of setting these bounds will be changed (see class docs for more details)
        """
        assert lower_bound is None or isinstance(lower_bound, Number), "Lower bound must beNone or a numeric type"
        assert upper_bound is None or isinstance(upper_bound, Number), "Upper bound must be None or a numeric type"
        new_full_bounds = lower_bound is not None and upper_bound is not None
        if new_full_bounds:
            assert lower_bound < upper_bound, f"Lower bound ({lower_bound}) is not less than upper bound ({lower_bound})"
            
        prev_min, prev_max = self.get_bounds()
        previous_exclusive_width = self.get_bound_width(True)
        
        # Set bounds and inclusivity
        self._bounds = tuple([None if b is None else self.dtype(b) for b in (self.dtype(lower_bound), self.dtype(upper_bound))])
        new_inclusive_lower, new_inclusive_upper = self._inclusivity
        if inclusive_lower_bound is not None:
            new_inclusive_lower = inclusive_lower_bound
        if inclusive_upper_bound is not None:
            new_inclusive_upper= inclusive_upper_bound
        self._inclusivity = (new_inclusive_lower, new_inclusive_upper)
        
        new_exclusive_width = self.get_bound_width() 
        is_new_width = not self.none_equal(new_exclusive_width, previous_exclusive_width) 
        is_new_inclusive_bounds = not self.none_equal(prev_min, self._bounds[0]) or not self.none_equal(prev_max, self._bounds[1]) 
        
        if not is_new_inclusive_bounds and not is_new_width:
            return
        
        # Check for incompatibilies
        if self.bound_is_finite():
            if new_exclusive_width  <= 0:
                raise ValueError(
                    f"With the dtype attribute and new inclusivity, the lower bound is as large as the the upper bound. \n " + 
                    "If using less precise float types, ensure that the bound is wide enough to account for exclusive values.")
            if self._fixed_width is not None:
                if new_exclusive_width < self._fixed_width:
                    self._fixed_width = new_exclusive_width
                    if is_new_inclusive_bounds and self._show_warnings:
                        warn("The new fixed width has been set to the width of the full bound (i.e the lower and upper bounds now represent a fixed width)")

        self._adjust_with_new_bound(previous_exclusive_width)

    def set_first_point_upper_bound(self, max_first_point: Union[Number, None], inclusive = True):
        """Set the upper bound of the first point
        
        Will not be set if both a last point upper bound and a fixed width are set
            - Causes a first point upper bound to be set automatically
                - (see Hierarchy and Fixed Width Example in class docs)
                
        Args:
            max_first_point (Union[Number, None]):
            inclusive (bool, optional): Whether or not the input represent an inclusive or exclusive bound. Defaults to True.
        """        
        assert(isinstance(max_first_point, (Number, None))), "'max_first_point' must be numeric or None"
        assert(isinstance(inclusive, (bool, np.bool_))), "'inclusive' must be a boolean"
        
        if self._bounds[1] and self._fixed_width:
            return
        if isinstance(max_first_point, Number):
            prev_max_first = self._max_first_point
            prev_inclusivity = self._max_first_inclusive
            max_first_point = self.dtype(max_first_point)
            if inclusive == prev_inclusivity and self.none_equal(prev_max_first, max_first_point):
                return
            
            if self._bounds[0] is not None:
                if max_first_point < self._bounds[0]:
                    max_first_point = self.get_calculable_lower_bound()
                    inclusive = True
                    if self._show_warnings:
                        warn(f"First point upper bound is less than the lower bound, therefore it is set to the lower bound value - i.e. a fixed first point")  
                
                # Bound is close enough to assume it should be a fixed start point
                elif self._bounds[0] <= max_first_point <= self._bounds[0] + self.eps:
                    max_first_point = self.get_calculable_lower_bound()
                    inclusive = True
                    
            # Max first point at at upper bound means the same as no bound
            if self._bounds[1] is not None and max_first_point >= self._bounds[1] - self.eps:
                max_first_point = None
                inclusive = True
                if self._show_warnings:
                    warn(f"First point upper bound is greater than or equal to the last point upper bound, therefore it is set to None.")

        elif prev_max_first is None:
            return
        
        self._max_first_point = max_first_point
        self._max_first_inclusive = inclusive
        self._adjust_with_new_first_last_bounds()
            

    def set_last_point_lower_bound(self, min_last_point: Union[Number, None], inclusive = True):
        """Set the lower bound of the last point
        
        Will not be set if both a first point lower bound and a fixed width are set
            - (last point lower bound set automatically, see Hierarchy and Fixed Width Example in class docs)

        Args:
            min_last_point (Union[Number, None]):
            inclusive (bool, optional): Whether or not the input represent an inclusive or exclusive bound. Defaults to True.
        """        
        assert(isinstance(min_last_point, (Number, None))), "'min_last_point' must be numeric or None"
        assert(isinstance(inclusive, (bool, np.bool_))), "'inclusive' must be a boolean"
        
        if self._bounds[0] and self._fixed_width:
            return
        if isinstance(min_last_point, Number):
            prev_min_last = self._min_last_point
            prev_inclusivity = self._min_last_inclusive
            min_last_point = self.dtype(min_last_point)
            if inclusive == prev_inclusivity and self.none_equal(prev_min_last, min_last_point):
                return
    
            if self._bounds[1] is not None:
                if min_last_point > self._bounds[1]:
                    min_last_point = self.get_calculable_upper_bound()
                    inclusive = True
                    if self._show_warnings:
                        warn(f"Last point lower bound is greater than the upper bound, therefore it is set to the upper bound value - i.e. a fixed last point.")  

                # Value is close enough to assume it should be a fixed end point
                elif self._bounds[1] - self.eps <= min_last_point <= self._bounds[1]:
                    min_last_point = self.get_calculable_upper_bound()
                    inclusive = True
                    
            # Min last point at at lower bound means the same as no bound
            if self._bounds[1] is not None and min_last_point <= self._bounds[0] + self.eps:
                min_last_point = None
                inclusive = True
                if self._show_warnings:
                    warn(f"Last point lower bound is less than or equal to the lower bound, therefore it is set to None (i.e no actual lower-bound on the last point).")
                    
        elif prev_min_last is None:
            return

        self._min_last_point = min_last_point
        self._min_last_inclusive = inclusive
        self._adjust_with_new_first_last_bounds()
        
    def set_min_separation(self, min_separation: Union[Number, None]):
        """
        Set the maximum distance between adjacent points - i.e. the min separation
            (this is value always assumed to be inclusive)
        
        Will raise an error if incompatible with a full bound width, a set fixed width, or 'min_separation' <= 0

        Ensures that: 
            - `minAdjacentDist` * `(maxPoints  - 1)` >=  `minWidth` if a min width exists
            - `minAdjacentDist` * `(minPoints - 1)` <= `maxWidth` if a max width exists 
  
        Args:
            min_separation (Union[Number, None]): _description_
        """        
        
        assert(isinstance(min_separation, (Number, None))), "'min_separation' must be numeric or None"
        
        if self.none_equal(self._min_separation, min_separation):
            return
        
        min_width, max_width = self.get_curr_width_bounds()
        if not min_width > self.eps:
            min_width = None
        if isinstance(min_separation, Number):
            _, max_first = self.get_first_point_bounds(with_all_exclusivity=True)
            min_last, _ = self.get_last_point_bounds(with_all_exclusivity=True)
            min_separation = self.dtype(min_separation)
            if self.none_equal(self._min_separation, min_separation):
                return
            assert min_separation > 0, f"'min_separation' must be greater than 0: {min_separation} was inputted"
            if max_width and min_separation > max_width:
                if self._fixed_width:
                    raise ValueError(f"'min_separation' must be less than or equal to the fixed width {self._fixed_width}: {min_separation} was inputted)")
                else:
                    raise ValueError(f"'min_separation' must be less than or equal to the bound width {max_width}: {min_separation} was inputted")
                
            # Check compatibility with the set max separation
            if not self.pos_inf_less_than(min_separation, self._max_separation) and not self.none_equal(min_separation, self._max_separation):
                min_separation = self._max_separation
                if self._show_warnings and not self._stricter_points:
                    warn(f"The min_separation input ({min_separation}) is greater than the maximum separation {self._max_separation}. It has been set to the maximum separation")

            # ensure `minAdjacentDist` * `(minPoints - 1)` <= `maxWidth` (severe conflict)
            elif max_width and self._point_bounds[0] and self.dtype(self._point_bounds[0] - 1) * min_separation > max_width:
                if self._stricter_points: # reduce min separation
                    min_separation = max_width / self.dtype(self._point_bounds[0] - 1)
                    if self._max_separation:
                        self._max_separation = min_separation
                else: # reduce min points
                    new_min_point = self.integer_dtype(max_width / min_separation + 1.0)
                    self._point_bounds = (new_min_point, None) if self._point_bounds[1] is None else (new_min_point, new_min_point)
                    
            #ensure `minAdjacentDist` * `(maxPoints - 1)` >= `minWidth`
            elif min_width is not None or (max_first is not None and min_last is not None and min_last - max_first > 0):
                if min_width is None:
                    min_width = min_last - max_first
                if self._point_bounds[1] and self._point_bounds[1] * min_separation < min_width:
                    if self._stricter_points: # increase min separation
                        min_separation = min_width / self.dtype(self._point_bounds[1] - 1)
                    else: # increase max points
                        new_max_point = self.integer_dtype(ceil(min_width / min_separation + 1.0))
                        self._point_bounds = (self._point_bounds[0],  new_max_point)     
                             
        self._min_separation = min_separation
        if (min_width or max_width) and not self._stricter_points:
            self._strict_separation_no_width_change()
 
    def set_max_separation(self, max_separation: Union[Number, None]):
        """ `
        
        Set the maximum distance between adjacent points - i.e. the max separation
            (this is value always assumed to be inclusive)
        
        Will raise an error if incompatible with a full bound width, a set fixed width, or 'max_separation' <= 0

        Ensures: 
            - `maxAdjacentDist` * `(minPoints - 1)` <=  `maxWidth` if max width exists
            - `maxAdjacentDist` * `(maxPoints - 1)` >=  `minWidth` if min width exists

        Args:
            max_separation (Union[Number, None])
        """        
        assert(isinstance(max_separation, (Number, None))), "'min_separation' must be numeric or None"
        if self.none_equal(self._max_separation, max_separation):
            return
        
        min_width, max_width = self.get_curr_width_bounds()
        if not min_width > self.eps:
            min_width = None
        if isinstance(max_separation, Number):
            max_separation = self.dtype(max_separation)
            if self.none_equal(self._max_separation, max_separation):
                return
            assert max_separation> 0, f"'max_separation' must be greater than 0: {max_separation} was inputted"
            if max_width and max_separation > max_width:
                if self._fixed_width:
                    raise ValueError(f"'max_separation' must be less than or equal to the fixed width {self._fixed_width}: {max_separation} was inputted)")
                else:
                    raise ValueError(f"'max_separation' must be less than or equal to the bound width {max_width}: {max_separation} was inputted")
            
            # Check compatibility with the set min separation 
            if not self.neg_inf_less_than(self._min_separation, max_separation) and not self.none_equal(self._min_separation, max_separation):
                max_separation = self._min_separation
                if self._show_warnings:
                    warn(f"The max_separation input ({max_separation}) is less than the minimum separation {self._min_separation}. It has been set to the same value")

            # ensure max_separation * (maxPoints - 1) >=  minWidth (severe conflict)
            elif min_width and self._point_bounds[1] and self._point_bounds[1] * max_separation < min_width:
                if self._stricter_points: # increase max separation
                    max_separation = min_width / self.dtype(self._point_bounds[1] - 1)
                    if self._min_separation:
                        self._min_separation = self._max_separation
                else: # increase max points
                    new_max_point = self.integer_dtype(ceil(min_width / max_separation + 1.0))
                    self._point_bounds = (None,  new_max_point) if self._point_bounds[0] is None else (new_max_point, new_max_point)

            # ensure max_separation * (min_points - 1) >=  minWidth
            elif max_width and self._point_bounds[0] and self.dtype(self._point_bounds[0] - 1) * max_separation > max_width:
                if self._stricter_points: # reduce max separation
                    max_separation = max_width / self.dtype(self._point_bounds[0] - 1)
                else: # reduce min points
                    new_min_point = self.integer_dtype(max_width / max_separation + 1.0)
                    self._point_bounds = (new_min_point, self._point_bounds[1])

        self._max_separation = max_separation
        if (min_width or max_width) and not self._stricter_points:
            self._strict_separation_no_width_change()

    def set_min_points(self, min_points: Union[Number, None]):
        """ Set the minimum number of points (this is value always assumed to be inclusive)
        
        Will raise an error if min_points < 2
        
        Ensures that:
            - `maxAdjacentDist` * `(minPoints - 1)` <= `maxWidth` 
            
                (therefore also `minAdjacentDist` * `(minPoints - 1)` <= `maxWidth`)

        Args:
            min_points (Union[Number, None])
        """        
        assert(isinstance(min_points, (Number, None))), "min_points must be a number or None"
        prev_min, prev_max = self._point_bounds()
        if self.none_equal(min_points, prev_min):
                return
        min_width, max_width = self.get_curr_width_bounds()
        if not min_width > self.eps:
            min_width = None
        if isinstance(min_points, Number):
            min_points = self.integer_dtype(min_points)
            assert min_points > 1, f"min_points must be at least 2: {min_points} was inputted"

            # ensure less than previous max points
            if self.pos_inf_less_than(min_points, prev_max):
                self._point_bounds = (prev_max, prev_max)
                if self._show_warnings:
                    warn(f"The inputted min points {min_points} is greater than the set max_points {prev_max}: min_points is set to the max points")
                
            # ensure `minAdjacentDist` * `(minPoints - 1)` <= `maxWidth`: (severe conflict)
            elif max_width and self._min_separation and self._min_separation * min_points > max_width:
                if self._show_warnings:
                    warn(f"min points {min_points} times min_separation {self._min_separation} is greater than max width {max_width}. This may cause all point and/or separation values to be equal after adjustment")
                if self._stricter_points: #reduce min separation
                    self._min_separation = max_width / self.dtype(min_points - 1)
                    if self._max_separation:
                        self._max_separation = self._min_separation
                else: # reduce_min_points
                    min_points = self.integer_dtype(max_width / self._min_separation + 1.0)
                self._point_bounds = (min_points, min_points) if prev_max else (min_points, None) 
            
            # ensure `maxAdjacentDist` * `(minPoints - 1)` <= `maxWidth`:
            elif max_width and self._max_separation and self._max_separation * min_points > max_width:
                if self._stricter_points: #reduce max separation
                    self._max_separation = max_width / self.dtype(min_points - 1)
                    if self._min_separation:
                        self._min_separation = min(self._min_separation, self._max_separation)
                else: # reduce_min_points
                    min_points = self.integer_dtype(max_width / self._max_separation + 1.0)
                self._point_bounds = (min_points, prev_max)
            else:
                self._point_bounds = (min_points, prev_max)
        else:
            self._point_bounds = (min_points, prev_max)

        if self._stricter_points and (min_width or max_width):
            self._strict_points_no_width_change()
    
    def set_max_points(self, max_points: Union[Number, None]):
        """
        Set the maximum number of points (this is value always assumed to be inclusive)
        
        Will raise an error if max_points < 2
        
        Ensures that: 
            - `minAdjacentDist` * `(maxPoints - 1)` >= `minWidth` if a min width exists
                
                (therefore also `maxAdjacentDist` * `(maxPoints - 1)` >= `minWidth`)
            
        Args:
            max_points (Union[Number, None])
        """        
        assert(isinstance(max_points, (Number, None))), "min_points must be a number or None"
        prev_min, prev_max = self.get_number_of_points_bounds()
        if self.none_equal(max_points, prev_max):
            return
        
        if isinstance(max_points, Number):
            max_points= self.integer_dtype(max_points)
            assert max_points > 1, f"max_points must be at least 2: {max_points} was inputted"
            max_width = None
            min_width = None
            if self._fixed_width is None:
                _, max_first = self.get_first_point_bounds(with_all_exclusivity=True)
                min_last, _ = self.get_last_point_bounds(with_all_exclusivity=True)
                if self.bound_is_finite():
                    max_width = self.get_bound_width()
                if max_first is not None and min_last is not None and min_last - max_first > 0:
                    min_width = min_last - max_first     
            else:
                max_width = self._fixed_width
                min_width = self._fixed_width

            # ensure greater than previous min points
            if self.neg_inf_greater_than(prev_min, max_points):
                self._point_bounds = (prev_min, prev_min)
                if self._show_warnings:
                    warn(f"The inputted max points {max_points} is greater than the set min_points {prev_min}, max points is set to the min points")
                max_points = prev_min
                
            # ensure `maxAdjacentDist` * `(maxPoints - 1)` >= `minWidth` (severe conflict)
            elif min_width and self._max_separation and self._max_separation * max_points < min_width:
                if self._show_warnings:
                    warn(f"max points {max_points} times max separation {self._max_separation} is less than min width {min_width}. This may cause all point and/or separation values to be equal after adjustment")
                if self._stricter_points: #increase max separation
                    self._max_separation = min_width / self.dtype(max_points - 1)
                    if self._min_separation:
                        self._min_separation = self._max_separation
                else: # increase_max_points
                    max_points = self.integer_dtype(min_width / self._max_separation + 1.0)
                self._point_bounds = (max_points, max_points) if prev_min else (None, max_points) 

            # ensure `minAdjacentDist` * `(maxPoints - 1)` >= `minWidth`
            elif min_width and self._min_separation and self._max_separation * max_points < min_width:
                if self._stricter_points: #increase min separation
                    self._min_separation = min_width / self.dtype(max_points - 1)
                    if self._max_separation:
                        self._max_separation = max(self._min_separation, self._max_separation)
                else: # increase_max_points
                    max_points = self.integer_dtype(min_width / self._min_separation + 1.0)
                self._point_bounds = (prev_min, max_points)
            else:
                self._point_bounds = (prev_min, max_points)
        else:
            self._point_bounds = (prev_min, max_points)
        
        if self._stricter_points and (min_width or max_width):
            self._strict_points_no_width_change()
    
    def set_fixed_width(self, width: Union[Number, None]):
        """ `
        
        Set a fixed width (this is value always assumed to be inclusive)
        
        Will raise an error if 'width' is greater than the inclusive bound width or 'width' < 0
        
        (Note, will make small adjustment if 'width' is greater than the exclusive bound width, but less than or equal to the inclusive bound width)
        
        Args:
            width (Union[Number, None]): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_
        """        
        assert(isinstance(width, (Number, None))), "'width' must be numeric or None"
        if isinstance(width, Number):
            width = self.dtype(width)
            assert width > 0, f"A fixed width must be greater than 0: {width}  was inputted"
            if self.bound_is_finite():
                inclusive_bound_width = self.get_bound_width(False)
                assert width <= inclusive_bound_width, f"The inputted width {width} is greater than the inclusive bound width {inclusive_bound_width}"
                exclusive_bound_width = self.get_bound_width()
                if width >= exclusive_bound_width:
                    width = exclusive_bound_width
                    
                # Override or set previous upper first point / lower last point bounds
                self._min_last_point = self._bounds[0] + width
                self._min_last_inclusive = self._inclusivity[0]
                self._max_first_point = self._bounds[1] - width
                self._max_first_inclusive = self._inclusivity[1]
                
            elif self._bounds[0] is not None: # Implicity know minimum last point
                self._min_last_point = self._bounds[0] + width
                self._min_last_inclusive = self._inclusivity[0]
                
                if self._max_first_point is not None: # Set upper bound
                    upper_bound = self._max_first_point + width
                    if upper_bound - self.get_calculable_lower_bound() < width: # incompatible 'max_first_point'
                        self._bounds = (self._bounds[0], self._min_last_point + self.eps)
                        self._inclusivity = (self._inclusivity[0], True) if not self._inclusivity[0] else (self._inclusivity[0], False) 
                        self._max_first_point = None
                    else:
                        self._bounds = (self._bounds[0], upper_bound)
                        self._inclusivity = (self._inclusivity[0], self._max_first_inclusive)
                    
            elif self._bounds[1] is not None: # Implicity know maximum first point
                self._max_first_point = self._bounds[1] - width
                self._max_first_inclusive = self._inclusivity[1]
                
                if self._min_last_point is not None: # Set lower bound
                    lower_bound = self._min_last_point - width
                    if lower_bound > self.get_calculable_upper_bound() - width: # incompatible 'min_last_point'
                        self._bounds = (self.get_calculable_upper_bound() - width, self._bounds[1]) 
                        self._inclusivity = (True, self._inclusivity[1])
                        self._min_last_point = None
                    else:
                        self._bounds = (lower_bound, self._bounds[1])
                        self._inclusivity = (self._min_last_inclusive, self._inclusivity[1])

            # No full bounds are set, but first upper and/or last lower bounds are 
            # (one or both of lower/upper bound can be determined)
            elif self._max_first_point is not None or self._min_last_point is not None: 
                set_bounds = False
                _, excl_max_first = self.get_first_point_bounds(with_all_exclusivity=True)
                excl_min_last, _= self.get_last_point_bounds(with_all_exclusivity=True)
                if excl_max_first is not None and excl_min_last is not None:
                        upper_bound = excl_max_first + width
                        lower_bound = excl_min_last - width
                        set_bounds = True
                        if upper_bound - lower_bound < width: # imcompatible bounds
                            diff = width - (upper_bound - lower_bound)
                            if diff == self.eps:
                                upper_bound += self.eps
                                self._max_first_point = self._max_first_point + self.eps
                                self._max_first_inclusive = not self._max_first_inclusive
                            else:
                                added = diff / self.dtype(2)
                                self._max_first_point = self._max_first_point + added 
                                self._min_last_point = self._min_last_point - added 
                                lower_bound -= added 
                                upper_bound += added
                        self._bounds = (lower_bound, upper_bound)
                        self._inclusivity= (self._min_last_inclusive, self._max_first_inclusive)
                if not set_bounds:
                    if self._max_first_point is not None: # Set upper bound
                        upper_bound = self._max_first_point + width
                        self._bounds = (self._bounds[0], upper_bound)
                        self._inclusivity = (self._inclusivity[0], self._max_first_inclusive)
                    if self._min_last_point is not None: # Set lower bound
                        lower_bound = self._min_last_point - width
                        self._bounds = (lower_bound, self._bounds[1])
                        self._inclusivity = (self._min_last_inclusive, self._inclusivity[1])
            
            previous_fixed_width = self._fixed_width
            self._fixed_width = width
            if self.none_equal(previous_fixed_width, self._fixed_width): # (for internal adjustment)
                return
            self._adjust_with_new_fixed_width()
        else:
            self._fixed_width = None
            return
    
    def all_bounds_set(self):
        bounds_set = self.bound_is_finite()
        min_points, max_points = self._point_bounds
        return bounds_set and all(var is not None for var in (self._max_first_point, self._min_last_point, self._min_separation, self._max_separation, min_points, max_points))
    
    def get_inclusivity(self) -> Tuple[bool, bool]:
        return self._inclusivity
    
    def get_bounds(self, with_exclusivity = False, with_half_exclusivity = False) -> tuple[Number | None, Number | None]:
        """ Retrieve the full bounds (i.e. minimum value of the first point, maximum value of the last point)
        
        Args:
            with_exclusivity (bool, optional): If True, adjusts values for exclusivity. Defaults to False.
                (see get_calculable_lower_bound() and get_calculable_upper_bound())
        """        
        if not with_exclusivity:
            return self._bounds
        if with_half_exclusivity:
            (self.get_calculable_lower_bound(), self._bounds[1])
        else:
            return (self.get_calculable_lower_bound(), self.get_calculable_upper_bound())
    
    def get_bound_width(self, with_exclusivity = True):
        """
        Get width of the x-bounds 
        
        If the bounds are infinite (one or both values are None), an numpy inf value is returned
        Args:
            with_exclusivity (bool, optional): If True, adjusts values for exclusivity. Defaults to True.
                (see get_calculable_lower_bound() and get_calculable_upper_bound())
        """        
        if not self.bound_is_finite():
            return np.inf
        if not with_exclusivity:
            return self._bounds[1] - self._bounds[0]  
        else:
            return self.get_calculable_upper_bound() - self.get_calculable_lower_bound()
        
    def get_first_point_bounds(self, with_all_exclusivity = False, with_half_exclusivity = True) -> tuple[Number | None, Number | None]:
        """
        Get bounds of the first point

        Args:
            with_exclusivity (bool, optional): If True, adjusts both values for exclusivity. Defaults to False.
                (see get_calculable_lower_bound() and get_calculable_upper_bound())
            with_half_exclusivity (bool, optional): If True, adjust the first value for exclusivity, but not the second. Defaults to True
        """
        if with_all_exclusivity:
            if self._max_first_inclusive:
                return (self.get_calculable_lower_bound(), self._max_first_point)
            else:
                return (self.get_calculable_lower_bound(), self._get_before(self._max_first_point))
        elif with_half_exclusivity:
            return (self.get_calculable_lower_bound(), self._max_first_point)
        else:
            return (self._bounds[0], self._max_first_point)
    
    def get_last_point_bounds(self, with_all_exclusivity = False, with_half_exclusivity = True) -> tuple[Number | None, Number | None]:
        """
        Get bounds of the last point

        Args:
            with_exclusivity (bool, optional): If True, adjusts values for exclusivity. Defaults to False.
                (see get_calculable_lower_bound() and get_calculable_upper_bound())
            with_half_exclusivity (bool, optional): If True, adjust the first value for exclusivity, but not the second. Defaults to True
        """
        if with_all_exclusivity:
            if self._min_last_inclusive:
                return (self._min_last_point, self.get_calculable_upper_bound())
            else:
                return (self._get_after(self._max_first_point), self.get_calculable_upper_bound())
        elif with_half_exclusivity and not self._min_last_inclusive:
            return (self._get_after(self._max_first_point), self._bounds[1])
        else:
            return (self._max_first_point, self._bounds[1])
    
    def get_curr_width_bounds(self) -> tuple:
        """
        Get current width bounds with the first/last point bounds, full bounds, and/or fixed width value

        Will return `None` if a maximum width has not been defined. 
        
        Will return `nfinfo(self.dtype).tiny` if a minimum not been defined
        """
        if self._fixed_width is not None:
            return self._fixed_width, self._fixed_width
        
        ex_max , ex_max_first = self.get_first_point_bounds(with_all_exclusivity=True)
        ex_min_last, ex_min = self.get_last_point_bounds(with_all_exclusivity=True)
        min_width = None
        max_width = None
        if ex_min_last is not None and ex_max_first is not None and ex_min_last > ex_max_first:
            min_width = ex_min_last - ex_max_first
        else:
            min_width = self.eps
        if self.bound_is_finite():
            max_width = ex_max - ex_min
            
        return (min_width, max_width)
    
    def get_number_of_points_bounds(self) -> tuple:
        """Get the minimum and maximum number of points (may be None values to indicate they haven't been set)
        """
        return self._point_bounds
    
    def get_calculable_lower_bound(self):
        """ 
        Get lower bound that is calculable (or None if no bound)
        Aka. if an exclusive lower_bound, get the value right after as defined by the 'dtype'. 
            Otherwise, return lower bound as is.
        """        
        if self._inclusivity[0]:
            return self._bounds[0]
        else:
            return self._get_after(self._bounds[0])
    
    def get_calculable_upper_bound(self):
        """Get upper bound that is calculable (or None if no bound)
        If an exclusive upper bound, get the value right before as defined by 'dtype'
            Otherwise, return upper bound as is."""
        if self._inclusivity[1]:
            return self._bounds[1]
        else:
            return self._get_before(self._bounds[1])
    
    def get_min_adjacent_distance(self):
        return self._min_separation
    
    def get_max_adjacent_distance(self):
        return self._max_separation
    
    def bound_is_finite(self) -> bool: 
        """Check if both bound values are set to a number (i.e a finite bound)"""
        return not (self._bounds[0] is None or self._bounds[1] is None)
    
    def in_inclusive_bound_range(self, val: Number):
        assert(issubclass(val, Number)), "Input must be numeric"
        return (self._bounds[0] is None or val >= self._bounds[0]) and (self._bounds[1] is None or val <= self._bounds[1])
    
    def in_exclusive_bound_range(self, val: Number):
        assert(issubclass(val, Number)), "Input must be numeric"
        lower_inclusive, upper_inclusive = self.get_inclusivity()
        if lower_inclusive and upper_inclusive:
            return self.in_inclusive_bound_range(val)
        exclusive_x_min = self.get_calculable_upper_bound()
        exclusive_x_max = self.get_calculable_lower_bound()
        return (self._bounds[0] is None or val >= exclusive_x_min) and (exclusive_x_max is None or val <= exclusive_x_max)
    
    def bound_str(self) -> str:
        out = "("
        out += "-inf, " if self._bounds[0] is None else f"{self._bounds[0]}, "
        out += "inf)" if self._bounds[1] is None else f"{self._bounds[1]}) "
        return out
    
    def _get_before(self, val):
        if val is None:
            return None
        return val - self.eps
    
    def _get_after(self, val):
        if val is None:
            return None
        return val + self.eps
    
    @staticmethod
    def none_equal(val1, val2):
        """ val1 == val2 operation
        
        Assumes val = `None` represents the same infinity for both values
        """
        return (val1 == val2) if (val1 is not None and val2 is not None) else (val1 is None and val2 is None)
    
    @staticmethod
    def none_min(val1, val2):
        """min(val1, val2): returns min or the value that is not None
        """
        return val2 if val1 is None else val1 if val2 is None else min(val1, val2)
    
    @staticmethod
    def none_max(val1, val2):
        """max(val1, val2): returns max or the value that is not None 
        """
        return val2 if val1 is None else val1 if val2 is None else max(val1, val2)

    @staticmethod
    def neg_inf_less_than(val1, val2) -> bool:
        """ val1 < val2  
        
        Assumes `None = -inf`
        """
        if val1 is not None and val2 is not None:
            return val1 < val2
        if val1 is None != val2 is None:
            return val1 is None
        return False
    
    @staticmethod
    def neg_inf_greater_than(val1, val2):
        """val1 > val2
        
        - Assumes `None = -inf` """
        if val1 is not None and val2 is not None:
            return val1 > val2
        if val1 is None != val2 is None:
            return val2 is None
        return False
    
    @staticmethod
    def pos_inf_less_than(val1, val2) -> bool:
        """ val1 < val2
        
        - Assumes `None = inf`
        """
        if val1 is not None and val2 is not None:
            return val1 < val2
        if val1 is None != val2 is None:
            return val2 is None
        return False
    
    @staticmethod
    def pos_inf_greater_than(val1, val2):
        """val1 > val2
        
        - Assumes `None = inf` """
        if val1 is not None and val2 is not None:
            return val1 > val2
        if val1 is None != val2 is None:
            return val1 is None
        return False
    
    def get_conditional_min_max_separation(self, num_points):
        """ Given a certain number of points, get the actual min and max separation given width bounds
        Useful when current min and max separation are the theoretical min and max or unknown
        """
        assert num_points > 1, "input must be greater than 1"
        
        denom = self.dtype(num_points - 1)
        if self._fixed_width is not None:
            out = self._fixed_width / denom 
            return out, out
        
        min_width, max_width = self.get_curr_width_bounds()
        output_min_separation = self._min_separation or self.eps
        if min_width > self.eps:
            output_min_separation = max(output_min_separation, min_width / denom)
        output_max_separation = None
        if max_width is not None:
            output_max_separation = min(max_width / denom, self._max_separation)
            output_min_separation = max(output_min_separation, output_max_separation)
        return output_min_separation, output_max_separation

    def get_conditional_min_max_points(self, separation):
        """ Given a certain separation, get the actual min and max points given width bounds
        Useful when current min and max points are the theoretical min and max or unknown
        """
        denom = self.dtype(separation)
        assert denom  > 0, "input must be greater than 0"
        
        min_width, max_width = self.get_curr_width_bounds()
        output_max_points = self._point_bounds[1]
        if max_width:
            output_max_points = self.none_min(self.integer_dtype(max_width / denom + 1.0), output_max_points)
            
        output_min_points = self._point_bounds[0] or self.integer_dtype(2)
        if min_width > self.dtype(output_min_points) * self.eps:
            output_min_points = max(self.integer_dtype(ceil(min_width / denom + 1.0)), output_min_points)
        output_min_points = self.none_min(output_min_points, output_max_points)
            
        return output_min_points, output_max_points
    
    def get_conditional_points_with_width(self, width):
        """ Given a certain width, get the actual min and max points 
            (given set separations and/or set points)
        """
        width = self.dtype(width)
        assert width  > 0, "input must be greater than 0"
        output_max_points = self._point_bounds[1]
        if self._min_separation is not None:
            output_max_points = self.none_min(self.integer_dtype(width / self._min_separation + 1.0), output_max_points)
        
        output_min_points = self._point_bounds[0] or self.integer_dtype(2)
        if self._max_separation is not None:
            output_min_points = max(self.integer_dtype(ceil(width / self._max_separation + 1.0)), output_min_points)
        return output_min_points, output_max_points
        
    def _adjust_with_new_bound(self, prev_max_width):
        """(internal: validate all other attribute given new full bounds,
        
        - assumes basic validity of inputs   
        - assumes it was directly called from set_full_bounds()
        """
        # in_min, in_max = self.get_bounds()
        ex_min, ex_max_first = self.get_first_point_bounds(with_all_exclusivity=True)
        ex_min_last, ex_max = self.get_last_point_bounds(with_all_exclusivity=True)
        prev_min_width, new_max_width = self.get_curr_width_bounds()
        if not prev_min_width > self.eps:
            prev_min_width = None
        
        # Check validity of points first/last point bound (i.e min width)
        changed_first_last_bounds = False
        if ex_max_first is not None:
            if ex_min is not None and ex_max_first <= ex_min + self.eps: 
                self._max_first_point = ex_min
                self._max_first_inclusive = True
                changed_first_last_bounds = True
                
            elif ex_max is not None and ex_max_first >= ex_max - self.eps:
                self.max_first_point = None
                changed_first_last_bounds= True
                
        if ex_min_last is not None:
            if ex_min is not None and ex_min_last <= ex_min + self.eps:
                self.min_last_point = None
                changed_first_last_bounds = True
                
            elif ex_max is not None and ex_min_last >= ex_max - self.eps:
                self.min_last_point = ex_max
                self._min_last_inclusive = True
                changed_first_last_bounds= True
    
        # (different requirements for fixed_width)
        if self._fixed_width is not None:
            if new_max_width is not None and self._fixed_width > new_max_width:
                self._fixed_width = new_max_width
            self._adjust_with_new_fixed_width(called_from_fixed_width=False)
            return 

        # See if there is greater/new min width
        min_width = None
        larger_min_width = False
        if changed_first_last_bounds:
            min_width, _ = self.get_curr_width_bounds()
            if not min_width > self.eps:
                min_width = None
            if prev_min_width is None:
                larger_min_width = min_width is not None
            elif min_width is not None:
                larger_min_width = min_width > prev_min_width
        
        # Check if there is a smaller/new max width     
        smaller_max_width = False
        if prev_max_width is None:
            smaller_max_width = new_max_width is not None 
        elif new_max_width is not None:
            smaller_max_width = prev_max_width > new_max_width

        # If any tighter bound could affect num points bounds / separation -> call adjustment
        if smaller_max_width or larger_min_width:
            self._adjust_with_valid_widths()
        
    
    def _adjust_with_new_first_last_bounds(self):
        """(internal: assumed to be called directly from setting new first upper / last lower point bounds)
        """
        if self._fixed_width is not None: # May need to change point bounds
            self.set_fixed_width(self._fixed_width)
 
        # Now assumes all point bounds / widths are valid
        new_min_width, _ = self.get_curr_width_bounds()
        if not new_min_width > self.eps:
            return
        self._adjust_with_valid_widths()

    def _adjust_with_new_fixed_width(self, called_from_fixed_width = True):
        """ (internal: assumed to be called from new full bound adjustment or from fixed_width setting)
        
       - Assumes fixed width with within full bounds
       - Assumes a new fixed width
        """
        
        if not called_from_fixed_width: # could be new first/last point bounds
            self.set_fixed_width(self._fixed_width)
        self._adjust_with_valid_widths()
  
    def _adjust_with_valid_widths(self):
        """(internal: assumes first / last point bounds and all width values are valid. 
        
        Called from tightening any first/last bound or width)"""
        
        min_points, max_points = self._point_bounds
        num_point_bounds = sum([var is None for var in (min_points, max_points)])
        num_separation_bounds = sum([var is None for var in (self._min_separation, self._max_separation)])
        if not num_point_bounds and not num_separation_bounds:
            return

        min_width, max_width = self.get_curr_width_bounds()
        if not min_width > self.eps:
            min_width = None
        if not min_width and not max_width:
            return # -> nothing bounding the min/max points or separations
        
        # No limits on points w/o separation bounds
        if self._stricter_points and not num_separation_bounds:
            self._strict_points_no_width_change()
            return
        if not num_separation_bounds:
            return
        
        # Check for all extreme cconflicts
        if max_width:
            if self._min_separation and self._min_separation > max_width: # most severe conflict
                if not self._stricter_points and min_width and min_width < max_width / 2:
                    self._max_separation = None
                    self._point_bounds = (None, None)
                    self._min_separation = min_width
                    if self._show_warnings:
                        warn(f"A width change occured, min_separation is now larger than the max width. Min separation has been set to an existing min width {min_width}, all other point / separation bounds set to None.")
                elif self._stricter_points and (min_points or max_points):
                    self._max_separation = None
                    self._min_separation = None
                    if self._show_warnings:
                        warn("A width change occured, min_separation is now larger than the max width. All separation bounds have been set to None")
                else:
                    self._point_bounds = (self.integer_dtype(2), None)
                    self._max_separation = max_width
                    self._min_separation = None
                    if self._show_warnings:
                        warn("A width change occured, min_separation is now larger than the max width. \n max_separation is set the max_width and min points to 2. All other points / separation bounds set to None (possibly tempoary)")
                    
            elif self._min_separation and min_points and self._min_separation * self.dtype(min_points-1) > max_width: # severe conflict
                if self._stricter_points and (min_points or max_points):
                    self._min_separation = None
                    self._max_separation = None
                    if self._show_warnings:
                        warn("A width change occured, now min separation times the min number of points is greater than the max width. \n All separation bounds have been set to None (possibly temporary)")
                else:
                    self._max_separation = None
                    self._point_bounds = (None, None)
                    if self._show_warnings:
                        warn("A width change occured, now min separation times the min number of points is greater than the max width. \n The min and max points, and max separation, have been set to None (possibly temporary)")
                    
            elif self._max_separation and self._max_separation > max_width: # severe conflict
                if self._stricter_points and (min_points or max_points):
                    self._max_separation = None
                    if self._show_warnings:
                        warn("A width change occured, now max_separation greater than the max_width. The max separation has been set to None")
                else:
                    self._min_separation = None
                    self._max_separation = max_width
                    self._point_bounds = (self.integer_dtype(2), None)
                    if self._show_warnings:
                        warn("A width change occured, now max_separation is greater than the max width. \n max_separation has been set to the max width, and min points to 2. All others point / separation bounds set to None")
                    
            elif self._max_separation and min_points and (self._max_separation * self.dtype(min_points-1) > max_width):
                if self._stricter_points:
                    self._max_separation = None
                else:
                    self._max_separation = max_width 
                    self._point_bounds = (self.integer_dtype(2), max_points)       
        if min_width:
            if self._max_separation and self._point_bounds[1] and self._max_separation * self.dtype(self._point_bounds[1]-1) < min_width: # most severe conflict
                if self._stricter_points:
                    self._min_separation = None
                    self._max_separation = None
                    if self._show_warnings:
                        warn("A width change occured, now max_separation times the max_points is smaller than the min width. The min and max separation have been set to None (possibly temporary)")
                else:
                    self._point_bounds = (None, None)
                    if self._show_warnings:
                        warn("A width change occured, now max_separation times the max_points is smaller than the min width. The min and max points have been set to None (possibly temporary)")
            elif self._min_separation and self._point_bounds[1] and self._min_separation * self.dtype(self._point_bounds[1]-1) < min_width:
                if self._stricter_points:
                    self._min_separation = None
                else:
                    self._point_bounds = (self._point_bounds[0], None)
                          
        num_point_bounds = sum([var is None for var in (self._point_bounds[0], self._point_bounds[1])])
        num_separation_bounds = sum([var is None for var in (self._min_separation, self._max_separation)])
        if not num_point_bounds and not num_separation_bounds: # all bounds were reset -> want at least one
            self._point_bounds = (2, None)

        if self._stricter_points and (self._point_bounds[0] or self._point_bounds[1]):
            self._strict_points_no_width_change()
        elif not self._stricter_points and (self._min_separation or self._max_separation):
            self._strict_separation_no_width_change()
    
    def _strict_points_no_width_change(self):
        """internal: check separations and try to find new ones
        
        - assumes valid widths and asserts "stricter_points"
        - assumes basic error checking of num point bounds occured
        - assumes 2 <= min_points <= max_points 
        """
        assert(self._stricter_points), "should have stricter points"
        min_points, max_points = self._point_bounds
        min_width, max_width = self.get_curr_width_bounds()
        if not min_width > self.eps:
            min_width = None
        
        if min_width and max_points: #Can determine min separation
            min_sep = (min_width / self.dtype(max_points - 1))
            self._min_separation = min_sep if self._min_separation is None else max(min_sep, self._min_separation)
        
        if max_width and min_points: #Can determine max separation
            max_sep = (max_width / self.dtype(min_points - 1))
            if self._min_separation:
                max_sep = max(self._min_separation, max_sep)
            self._max_separation = max_sep if self._max_separation is None else min(max_sep, self._max_separation)
        
    def _strict_separation_no_width_change(self):
        """internal: check num point bounds try to find new ones
 
        assumes valid widths and asserts not "stricter_points"
        assumes basic error checking of num point bounds and separations already occured
        assumes self._min_separation <= self._max_separation <= max_width
        """
        assert(not self._stricter_points), "should have stricter separations"
        min_points, max_points = self._point_bounds
        min_width, max_width = self.get_curr_width_bounds()
        if not min_width > self.eps:
            min_width = None
        
        true_min_points = self._point_bounds[0]
        true_max_points = self._point_bounds[1]
        if min_width and self._max_separation: # Can determine min points
            min_points = self.integer_dtype(ceil(min_width / self._max_separation + 1.0))
            true_min_points = min_points if true_min_points is None else max(min_points, true_min_points)

        if max_width and self._min_separation: #Can determine max points
            max_points = self.integer_dtype((max_width / self._min_separation + 1.0))
            if true_min_points:
                max_points = max(true_min_points, max_points)
            true_max_points = max_points if true_max_points is None else min(max_points, true_max_points)
        self._point_bounds = (true_min_points, true_max_points)