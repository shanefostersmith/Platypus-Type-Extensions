import numpy as np
from functools import partial
from collections.abc import Hashable
from numbers import  Number
from typing import Literal, Union, Optional
from .point_bounds import PointBounds, BoundsState
# from ._bounds_tools import BoundsState
from ._distribution_tools import sym_bound_adjustment
from .real_bijection import RealBijection

class SymmetricBijection(RealBijection):
    """ `
    **A class for creating symmetric bijections between two real valued domains**.
    
    Ouput alues are mapped from functions that are monotonic increasing/decreasing on both "sides" of a global minima or maxima. 
    
    Functions that fit this description include unimodal distributions and functions with *convexity*.
    For example, a normal/Gaussian distribution is *unimodal distriution* and `y = x^2` is a *concave* upward function.
    
    You may also provide any strictly monotonic function and this class will create a symmetric function around the "center" x.
    - For example, say you provide  `y = e^x` and a center x value of 0. 
    - This class would then describe a concave upward function where the "right" half the output arrays are from y = e^x (where x >= 0) 
        and the "left" half are from y = e^(-x) (where x <= 0)
    
    Unimodal Distributions include:
    - *Normal/Gaussian*
    - *Cauchy*
    - *Logistic*
    - *Laplace*
    - *Hyperbolic secant*
    - *Inverse Quadratic Function* 
    - *Quadratic, Cosine, Kaiser-Bessel Kernels*
    - and more
    
    Concave-Upward Functions include:
    - *Quadratic* 
    - *Even-powered polynomials*
    - *Hyperbolic cosine*
    - *Log Cosh*
    - and more
    

    This is a subclass of `RealBijection` and uses most of the same logic and methods.
    
    Note that, **only static PointBounds are currently supported** (no setting of bounds after initialization)
    """    
    
    def __init__(self,
        forward_function: Union[partial, callable],
        center_x: Number,
        right_side_provided: bool,
        min_width: Number,
        max_width: Number,
        min_points: int = 4,
        max_points: int = 100,
        inverse_function: Union[partial, callable, None] = None,
        forward_args: dict = {},
        inverse_args: dict = {},
        include_global_extrema = False,
        exclude_global_extrema = False,
        point_bounds: Union[PointBounds, BoundsState, None] = None,
        precision: Literal['double','single'] = 'double',
        unique_id: Optional[Hashable] = None):
        """ 
        Args:
            forward_function (partial | callable): The function from x values to y values 
                - Only required to be valid *for one side* of the global minima or maxima
                - (See *RealBijection* for more details)
            center_x (float): The x value that maps to the global minima / maxima
            right_side_provided (bool): 
                - If True, indicates that the forward / inverse functions are for x values >= center x 
                - If False, indicates that the forward / inverse functions are for x values <= center x 
            min_width (float): The minimum distance between first and last **x** value 
                - i.e. the minimum distance from the center x **times 2**
            max_width (float): The maximum distance between first and last **x** value 
                - i.e. the maximum distance from the center x **times 2**
            min_points (int, optional): the minimum number of points in any output distribution
            max_points (int, optional): the maximum number of points in any output distribution
            forward_args (dict, optional): Keyword arguments for the forward map function
            inverse_map (partial | callable | None): The inverse function of the forward map
                - The function from y values to x values *for one side* of the global minima or maxima
                - See (*RealBijection* for more details on inverse functions)
            inverse_args (dict | None): Keyword arguments for the inverse map function
            include_global_extrema (bool, optional): Indicates if the global minima or maxima should **always** be included in the output arrays. Defaults to False.
                - If True, this implies all output arrays will be odd-sized
            exclude_global_extrema (bool, optional): Indicates if the global minima or maxima should **always** be excluded in the output arrays. Defaults to False.
                - If True, this implies all output arrays will be even-sized
                - If *include_global_extrema* is False and *exclude_global_extrema* is False, 
                    then the global minima/maxima may or may not be included in output arrays 
            point_bounds (PointBounds | BoundsState | None, optional): Defaults to None
                - If None, a PointBounds object will be created. If provided, the min_width/max_width and min_points/max_points are still required to be inputed.
                - Used to prevent multiple PointBounds objects from being created when using the same bounds across multiple symmetric bijections
                - **Should be created with the `create_bounds_for_symmetry()` function**
            precision (type, optional): The numeric type of the output numpy arrays, 'single' (float32) or 'double' (float64). Defaults to 'double`.
            unique_id (Hashable | None, optional): (See *RealBijection*)). Defaults to None.
        """        
        
        all_bounds = point_bounds
        if point_bounds is None or not isinstance(point_bounds, PointBounds):
          all_bounds = create_bounds_for_symmetry(
              center_x, 
              min_width, max_width, 
              min_points, max_points,
              include_global_extrema, 
              exclude_global_extrema,
              right_side_provided, 
              precision,
              create_bounds_state=True)
        else:
            all_bounds = point_bounds

        super(SymmetricBijection, self).__init__(
            forward_function, 
            x_point_bounds=all_bounds,
            inverse_function=inverse_function,
            fixed_forward_keywords=forward_args,
            fixed_inverse_keywords=inverse_args,
            unique_id=unique_id
        )
        self.center_x = center_x
        min_points, max_points, include_global_extrema, exclude_global_extrema = _fixed_mod_points(min_points, max_points, include_global_extrema, exclude_global_extrema)
        if min_points == max_points and  min_points % 2 == 0:
            assert exclude_global_extrema and not include_global_extrema
        elif min_points == max_points:
            assert not exclude_global_extrema and include_global_extrema
        
        self.include_global_extrema = include_global_extrema
        self.exclude_global_extrema = exclude_global_extrema
        self.right_side_provided = right_side_provided
        self._full_point_bounds = (min_points, max_points)
        self._full_width_bounds = (min_width, max_width)
        self._full_separation_bounds = (min_width / all_bounds.dtype(max_points - 1), max_width / all_bounds.dtype(min_points - 1))
    
    def _inclusive_adjustment(
        self, 
        output_start,
        output_points, 
        output_separation):
        
        full_max_points = self._full_point_bounds[1]
        full_min_sep = self._full_separation_bounds[0]
        double_output_points = (
            2*output_points + 1 
            if 2*self.point_bounds.max_points < full_max_points or full_min_sep < self.point_bounds.min_separation
            else 2*output_points - 1
        )
        curr_full_width = output_separation * self.point_bounds.dtype(double_output_points - 1)
 
        double_output_points, output_separation = sym_bound_adjustment(
            double_output_points, output_separation, curr_full_width,
            self._full_point_bounds, self._full_separation_bounds, self._full_width_bounds,
            self.point_bounds.dtype, even_points = False
        )
        output_points = double_output_points // 2 + 1
        if not self.right_side_provided:
            output_start = self.center_x
            output_separation = -output_separation
        
        return output_start, output_points, output_separation
            
    def _exclusive_adjustment(
        self,
        output_start,
        output_points, 
        output_separation, 
    ):
        
        full_max_points = self._full_point_bounds[1]
        full_min_sep = self._full_separation_bounds[0]
        double_output_points = 2*output_points
        if 2*self.point_bounds.max_points < full_max_points or full_min_sep < self.point_bounds.min_separation:
            double_output_point += 2

        curr_full_width = output_separation * self.point_bounds.dtype(double_output_points - 1)
        double_output_points, output_separation = sym_bound_adjustment(
            double_output_points, output_separation, curr_full_width,
            self._full_point_bounds, self._full_separation_bounds, self._full_width_bounds,
            self.point_bounds.dtype, even_points = True
        )
        # remove center point, find new start or end
        half_separation = output_separation  / 2.0 
        output_points = double_output_points // 2
        if self.right_side_provided:
            output_start = self.center_x + half_separation
        else:
            output_start = self.center_x - half_separation
            output_separation = -output_separation
        return output_start, output_points, output_separation

    def _choice_adjustment(
        self,
        output_start,
        output_points, 
        output_separation, 
    ):
        
        full_min_points, full_max_points = self._full_point_bounds
        full_min_separation, full_max_separation = self._full_separation_bounds
        assert full_min_points < full_max_points
        
        # choose whether to include center or not
        double_output_points = 2*output_points
        include_center = 0
        min_room = full_min_separation - self.point_bounds.min_separation 
        max_room = self.point_bounds.max_separation - full_max_separation 
        if min_room == 0 and max_room == 0 and output_points % 2 == 1:
            include_center = 1
            double_output_points = double_output_points - 1 if double_output_points > full_max_points else double_output_points + 1
        elif min_room > 0 and output_separation < full_min_separation:
            include_center = 1
            double_output_points -= 1
            output_separation = full_min_separation
        elif max_room > 0 and output_separation > full_max_separation:
            include_center = 1
            double_output_points += 1
            output_separation = full_max_separation
        elif min_room < 0 or max_room < 0: # can't reach all possible separations inside evolutions/mutations
            min_outer = 0 if min_room > 0 else abs(min_room)
            max_outer = 0 if max_room > 0 else abs(max_room)
            outer_room = min_outer + max_outer
            inner_room = self.point_bounds.max_separation - self.point_bounds.min_separation
            if not inner_room or (abs(min_outer + max_outer) / self.point_bounds.max_separation - self.point_bounds.min_separation < 0.05):
                include_center = output_points % 2
            else:
                include_center = int(np.random.choice([0, 1], p = np.array([inner_room, outer_room] / (outer_room + inner_room))))
            if include_center:
                prev_full_width = output_separation * self.point_bounds.dtype(double_output_points - 1)
                if min_outer == 0: # push toward max
                    double_output_points -= 1
                elif max_outer == 0: # push toward min
                    double_output_points += 1
                else:
                    double_output_points = (
                        double_output_points + 1 
                        if max_outer == 0 or abs(output_separation - full_max_separation) > abs(output_separation - full_min_separation)
                        else double_output_points - 1
                    )
                output_separation = prev_full_width / self.point_bounds.dtype(double_output_points - 1)

        curr_full_width = output_separation * self.point_bounds.dtype(double_output_points - 1)
        double_output_points, output_separation = sym_bound_adjustment(
            double_output_points, output_separation, curr_full_width,
            self._full_point_bounds, self._full_separation_bounds, self._full_width_bounds,
            self.point_bounds.dtype, even_points = not bool(include_center)
        )
        output_points = double_output_points // 2 + include_center
        if not self.right_side_provided:
            output_separation = -output_separation
        output_start = self.center_x if include_center else self.center_x + output_separation / 2.0
            
        return output_start, output_points, output_separation, include_center
        
    def _pre_distribution(self, output_start, output_points, output_separation):
        """
        Adjust start such that separation across the center is the same as the separation between the other points.
        
        Resolves differences between half bounds and full bounds
        
        returns (output_start, output_points, output_separation, "if center is included" (int))
        """        
        
        full_min_separation, full_max_separation = self._full_separation_bounds
        full_min_points, full_max_points = self._full_point_bounds
        if full_min_separation == full_max_separation:
            output_separation = full_min_separation
        if full_min_points == full_max_points:
            output_points = full_min_points
        
        curr_full_width = self.point_bounds.dtype(output_points - 1)*output_separation
        output_start = output_start if self.right_side_provided else self.center_x - curr_full_width
        include_center = 1
        if self.include_global_extrema:
            output_start, output_points, output_separation = self._inclusive_adjustment(
                output_start, output_points, output_separation,
            )
        elif self.exclude_global_extrema:
            include_center = 0
            output_start,output_points, output_separation = self._exclusive_adjustment(
                output_start, output_points, output_separation,
            )
        else:
            output_start, output_points, output_separation, include_center = self._choice_adjustment(
                output_start, output_points, output_separation, 
            )
        return output_start, output_points, output_separation, include_center

    def create_distribution(self, start_x, num_points, separation, **kwargs):
    
        start_x, num_points, separation, include_center = self._pre_distribution(start_x, num_points, separation)
        full_points = num_points*2 - include_center
        full_distribution = np.empty(full_points, self._return_type)

        if self.right_side_provided:
            full_distribution[num_points-include_center:] = super(SymmetricBijection, self).create_distribution(
                start_x, num_points, separation, reverse_x = False
            )
            full_distribution[:num_points-include_center] = full_distribution[num_points:][::-1]
        else:
            full_distribution[:num_points] = super(SymmetricBijection, self).create_distribution(
                start_x, num_points, separation, reverse_x = True
            )
            full_distribution[num_points:] = full_distribution[:num_points - include_center][::-1]
        return full_distribution

def _fixed_mod_points(
    min_points,
    max_points,
    include_global_extrema,
    exclude_global_extrema 
):
    if min_points == max_points and max_points % 2 == 1:
        if exclude_global_extrema:
            min_points += 1
            max_points += 1
        else:
            include_global_extrema = True
    elif min_points == max_points:
        if include_global_extrema:
            min_points += 1
            max_points += 1
        else:
            exclude_global_extrema = True  
    elif (exclude_global_extrema or include_global_extrema):
        if min_points + 1 == max_points:
            max_points += 1
        if include_global_extrema:
            if min_points % 2 == 0:
                min_points += 1
            if max_points % 2 == 0:
                max_points += 1
        elif exclude_global_extrema:
            if min_points % 2 == 1:
                min_points += 1
            if max_points % 2 == 1:
                max_points += 1
                
    return min_points, max_points, include_global_extrema, exclude_global_extrema
    
        
def create_bounds_for_symmetry( 
    center_x: Number,
    min_width: Number,
    max_width: Number,
    min_points: Number = 4,
    max_points: Number = 100,
    include_global_extrema = False,
    exclude_global_extrema = False,
    right_side_provided = True,
    precision: Literal['double','single'] = 'double',
    create_bounds_state = False):
    """
    Create a PointBounds object for a SymmetricBijection. The output PointBounds describe the bounds for half of the output distributions.
    The 'center_x' point is fixed when encoding, and potentially removed during creation of the output distribution
    
    This function is called at initialization by SymmetricBijection if a PointBounds object is not provided.
    
    Note, bounds are not exact due to rounding of number of points / 2. 
    
    See SymmetricBijection
    """    
    return_type = np.float64 if precision == 'double' else np.float32
    
    # Check min / max width and points
    min_width = return_type(min_width) 
    max_width = return_type(max_width)
    if min_width <= 0:
        raise ValueError("min_width must be > 0")
    
    if min_width > max_width:
        raise ValueError("min_width cannot be larger than max_width")
    min_width = return_type(
        max(2.0 * np.spacing(abs(center_x) + max_width / 2, dtype = return_type), min_width)
    )
    max_width = return_type(max(min_width, max_width))
    
    if min_points < 4:
        raise ValueError("min_points must be at least 4")
    if min_points > max_points:
        raise ValueError("min_point cannot be larger than the max_point")
    if include_global_extrema and exclude_global_extrema:
        raise ValueError("Only one of *include_global_extrema* or *exclude_global_extrema* can be True")
    
    min_points, max_points, include_global_extrema, exclude_global_extrema = _fixed_mod_points(min_points, max_points, include_global_extrema, exclude_global_extrema)

    # Determine "half" bounds
    true_min_separation = min_width / return_type(max_points - 1)
    true_max_separation = max_width / return_type(min_points - 1)
    half_min_points = min_points // 2 
    half_max_points = max_points // 2 
    if half_min_points % 2 == 0:
        half_min_points += 1
    if half_max_points % 2 == 0:
        half_max_points += 1
    half_max_width = max_width / 2.0
    half_min_width = min_width / 2.0
    
    assert half_min_points > 2
    x_min = center_x if right_side_provided else center_x - half_max_width
    x_max = center_x + half_max_width if right_side_provided else center_x
    point_bounds = PointBounds(
            x_min, x_max,
            minimum_points = half_min_points, 
            maximum_points = half_max_points,
            precision=precision
        )
    
    if min_width == max_width:
        point_bounds.set_fixed_width(x_max - x_min)
        point_bounds.set_max_separation(true_max_separation)
        if  point_bounds.min_separation > true_min_separation:
            point_bounds.set_max_points(half_max_points + 1)
        
    else:
        max_first_point = x_min if right_side_provided else x_max - half_min_width
        min_last_point = x_min + half_min_width if right_side_provided else x_max
        point_bounds.set_first_point_upper_bound(max_first_point)
        point_bounds.set_last_point_lower_bound(min_last_point)
        
        point_bounds.set_max_separation(true_max_separation)
        if point_bounds.min_separation > true_min_separation:
            point_bounds.set_max_points(half_max_points + 1)
        else:
            point_bounds.set_max_points(half_max_points)
        
    return point_bounds if not create_bounds_state else point_bounds.create_bounds_state()
