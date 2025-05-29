import numpy as np
from functools import partial
from collections.abc import Hashable
from numbers import Integral, Number
from .point_bounds import PointBounds
from ._bounds_tools import BoundsState
from .real_bijection import RealBijection

class SymmetricBijection(RealBijection):
    """ `
    (IN PROGRESS)
    
    A class for evolving and mutating **symmetric** arrays of real values.
    These values are mapped from distributions or functions that are monotonic increasing/decreasing on both "sides" of a global minima or maxima. 
    
    Functions that fit this description include unimodal distributions and functions with *convexity*:
        
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
    """    
    
    def __init__(self,
        forward_function: partial | callable,
        forward_args: dict | None,
        inverse_function: partial | callable | None,
        inverse_args: dict | None,
        convexity: bool,
        center_x: Number,
        right_side_provided: bool,
        min_width: Number,
        max_width: Number,
        min_points: int = 4,
        max_points: int = 100,
        include_global_extrema = False,
        exclude_global_extrema = False,
        point_bounds: PointBounds | BoundsState | None = None,
        return_type: type = np.float32,
        unique_id: Hashable | None = None):
        """ `
        Args
        ---
            forward_function (partial | callable): The function from x values to y values 
            
                - Only required to be valid *for one side* of the global minima or maxima
            
                - (See *RealBijection* for more details)
            
            forward_args (dict | None): Keyword arguments for the forward map function
            
            inverse_map (partial | callable | None): The inverse function of the forward map
                
                - The function from y values to x values *for one side* of the global minima or maxima
            
                - See (*RealBijection* for more details on inverse functions)
            
            inverse_args (dict | None): Keyword arguments for the inverse map function
            
            convexity (bool): Indicates if the function/distribution is concave downward or concave upwards
            
                - If True, then the function is strictly decreasing around a global maxima 
                (like a Normal/Gaussian distribution)
            
                - If False, then the function is strictly increasing around a global minima 
                (like `y = x^2`)
            
            center_x (Number: The x value that maps to the global minima / maxima
            
            right_side_provided (bool): 
            
                - If True, indicates that the forward / inverse functions are for x values > center x 
                
                - If False, indicates that the forward / inverse functions are for x values < center x 
            
            min_width (Number, optional): The minimum distance between first and last **x** value 

                - i.e. the minimum distance from the center x **times 2**
                
                - Must be provided if *point_bounds* is not provided
            
            max_width (Number, optional): The maximum distance between first and last **x** value 

                - i.e. the maximum distance from the center x **times 2**
                
                - Must be provided if *point_bounds* is not provided
                
            inclusive_min_width (bool, optional): Indicates if the min width is inclusive. Defaults to True
            
            inclusive_max_width (bool, optional): Indicates if the max width is inclusive. Defaults to True
                
            include_global_extrema (bool, optional): Indicates if the global minima or maxima should **always** be included in the output arrays. Defaults to False.
            
                - If True, this implies all output arrays will be odd-sized
            
            exclude_global_extrema (bool, optional): Indicates if the global minima or maxima should **always** be excluded in the output arrays. Defaults to False.

                - If True, this implies all output arrays will be even-sized
                
                - If *include_global_extrema* is False and *exclude_global_extrema* is False, 
                    then the global minima or maxima may or may not be included in output arrays 
            
            point_bounds (PointBounds, optional): Defaults to None
            
                - If None, a PointBounds object will be created 
            
                - Used to prevent multiple PointBounds objects from being created
                    (if using the same x width, center x, points bounds for multiple SymmetricUnimodal objects)
            
                - **Should be created with the 'create_bounds_for_symmetry()'* function**

            return_type (type, optional): The numeric type of the output numpy arrays. Defaults to numpy.float64.
            
                - *Currently, only numpy floats are supported (and not float128)
            
            unique_id (Hashable | None, optional): (See *RealBijection*)). Defaults to None.
        
        """        
        
        all_bounds = point_bounds
        if point_bounds is None or not isinstance(point_bounds, PointBounds):
          all_bounds, return_type = create_bounds_for_symmetry(
              center_x, min_width, max_width, 
              min_points, max_points,
              include_global_extrema, exclude_global_extrema,
              right_side_provided, return_type)
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
        self.include_global_extrema = include_global_extrema
        self.exclude_global_extrema = exclude_global_extrema
        self.right_side_provided = right_side_provided 

        
    def _adjust_start(self, output_start, output_points, output_separation):
        """
        Adjust start such that separation across the center is the same as the separation between the other points
            
        
        returns (output_start, output_points, output_separation, "if center is included" (int))
        """        
        flt2 = self._return_type(2)
        width = ((output_points - 1.0) * output_separation)
        x_bounds = self.point_bounds
        if self.right_side_provided:
            if self.include_global_extrema:
                return self.center_x, output_points, output_separation, 1
            elif self.exclude_global_extrema or output_start >= x_bounds.min_separation / flt2:
                output_start = output_separation / flt2 
                if output_start + width >  x_bounds.upper_bound:
                    output_points -= 1
                return output_start, output_points, output_separation, 0
            else:
                if self.center_x + width < x_bounds.min_last_point:
                    output_points += 1
                return self.center_x, output_points, output_separation, 1
        else:
            output_end = output_start + width
            if self.include_global_extrema:
                return self.center_x - width, output_points, output_separation, 1
            elif self.exclude_global_extrema or self.center_x - output_end >= x_bounds.min_separation / flt2:
                output_end = self.center_x - output_separation / flt2
                output_start = output_end - width
                if output_start < x_bounds.lower_bound:
                    output_start += output_separation
                    output_points -= 1
                return output_start, output_points, output_separation, 0
            else:
                output_start = self.center_x - width
                if output_start > x_bounds.max_first_point:
                    output_start -= output_separation
                    output_points += 1
                    return output_start, output_points, output_separation, 1

    def create_distribution(self, start_x, num_points, separation, **kwargs):
    
        start_x, num_points, separation, inclusive_center = self._adjust_start(start_x, num_points, separation)
        full_points = num_points*2 - inclusive_center
        full_distribution = np.empty(full_points, self._return_type)

        if self.right_side_provided:
            full_distribution[num_points-inclusive_center:] = super(SymmetricBijection, self).create_distribution(
                start_x, num_points, separation, False
            )
            full_distribution[:num_points-inclusive_center] = full_distribution[num_points:][::-1]
        else:
            full_distribution[:num_points] = super(SymmetricBijection, self).create_distribution(
                start_x, num_points, separation, False
            )
            full_distribution[num_points:] = full_distribution[:num_points - inclusive_center][::-1]
        return full_distribution
    
        
def create_bounds_for_symmetry( 
    center_x: Number,
    min_width: Number,
    max_width: Number,
    min_points: Number = 4,
    max_points: Number = 100,
    include_global_extrema = False,
    exclude_global_extrema = False,
    right_side_provided = True,
    return_type: type = np.float32,
    create_bounds_state = False):
    """
    Create a PointBounds object for a SymmetricUnimodal
    
    This function is called at initialization by SymmetricUnimodal if a PointBounds object is not provided.
    
    You can use this function to prevent multiple copies of a PointBounds object from being created
        (if creating multiple SymmetricBijection objects with the same 'x' bounds and min/max point bounds)
        
    See SymmetricBijection
    """    
    
    # Check return type
    if return_type is float:
        return_type = np.float32
    elif not issubclass(return_type, np.number):
        raise ValueError("The return type must be a numpy numeric type or a python float")
    
    # Check min / max width and points
    min_width = return_type(min_width) 
    max_width = return_type(max_width)
    if min_width <= 0:
        raise ValueError("min_width must be > 0")
    if min_width > max_width:
        raise ValueError("min_width cannot be larger than max_width")
    if min_points < 4:
        raise ValueError("min_points must be at least 4")
    if min_points > max_points:
        raise ValueError("min_point cannot be larger than the max_point")
    if include_global_extrema and exclude_global_extrema:
        raise ValueError("Only one of *include_global_extrema* or *exlcude_global_extrema* can be True")
    
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

        # Determine "half" bounds
        min_separation = min_width / return_type(max_points - 1)
        max_separation = max_width / return_type(min_points - 1)
        x_min = None
        x_max = None
        max_first_x = None
        min_last_x = None
        if right_side_provided:
            x_min = center_x + min_separation / 2 if exclude_global_extrema else center_x
            if include_global_extrema:
                max_first_x = x_min
            else:
                max_first_x = center_x + max_separation / 2
            x_max = center_x + max_width / 2
            min_last_x = center_x + (min_width / return_type(2))

        else:
            x_max = center_x - min_separation / 2 if exclude_global_extrema else center_x
            if include_global_extrema:
                min_last_x = x_max
            else:
                min_last_x  = center_x - max_separation / 2
            x_min = center_x - max_width / 2
            max_first_x = center_x - (min_width / return_type(2))
        
        half_min_points = min_points // 2
        half_max_points = max_points // 2
        if include_global_extrema:
            half_min_points += 1
            half_max_points += 1
        elif not exclude_global_extrema and max_points % 2 == 1:
            half_max_points += 1
        
        # Set bounds object
        point_bounds = PointBounds(
            x_min, x_max,
            min_points = half_min_points, 
            max_points = half_max_points,
            dtype = return_type
        )
        if min_width == max_width and include_global_extrema:
            point_bounds.set_fixed_width(x_max - x_min)
        else:
            point_bounds.set_first_point_upper_bound(max_first_x)
            point_bounds.set_last_point_lower_bound(min_last_x)
            
        point_bounds.set_min_separation(min_separation)
        point_bounds.set_max_separation(max_separation)
        
        if create_bounds_state:
            return point_bounds.create_bounds_state()
        return point_bounds
    