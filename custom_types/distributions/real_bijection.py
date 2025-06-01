import numpy as np
import inspect
import uuid
from warnings import warn
from functools import partial
from typing import Union, Tuple, Any
from collections.abc import Hashable
from numbers import Integral, Number
from math import ceil
from .point_bounds import PointBounds
from ._bounds_tools import BoundsState, _cascade_from_global

 
class RealBijection:
    """ 
    `
    **A class for creating bijections between two sets of real values**.
    
    This class is used for evolving and mutating sets of real `x` numbers that map to `y = f(x)` values, and the parameters that define that mapping, as a *single* optimization variable. 
    
    The most common use case would be evolve points along a distribution, where evolving each parameter of that distribution should depend on, or account for, the other parameters of that distribution. 
    In order to encode, decode and evolve these distributions, it is generally required that some one-to-one mapping – a *bijection* – be defined for the `x` and `y = f(x)` values. 
    Even if the `x` and `y = f(x)` values do not *directly* have a one-to-one mapping, combining *RealBijection* objects, and other techniques, can be used to design that one-to-one mapping.
    
    This class helps design these mappings, bound their outputs, store their meta data and provides functionality to use them in mutation and evolution methods.
        - See **init()** for details on how to format inputs and attributes.
    
    **A general definition of a *real bijection*** (and the definition used by this class):
    
    - A *mapping* **{`F`[`x`] -> `y`, `G`[`y`] -> `x`}** between a set of real values `X` and real set of values `Y`  where:
        -   `F` is a function from  `x` values to `y` values, and is the inverse of `G`
        -   `G` is a function from `y` values to `x` values, and is the inverse of `F`
        -   `F`[`xi`] `=` `yi`  *if and only if* `G`[`yi`] `=` `xi` for all `x` and `y`
                -  i.e. one `x` value *must*  map to exactly one `y` value, and vice versa 
    
    Attribute Descriptions
    ----
    *Required*:
        **forward_map**: A callable where the first positional argument is for input 'x' values
            - The 'forward' mapping function `F`[`x`] -> `y`
        
        **x_point_bounds**: A PointBounds or BoundsState object
            - Contains all `x` bounds and other bounds regarding how sets of points may be created (see `PointBounds`)
            - If it is a PointBounds class, it contains all `set` methods for changing `x` bounds
            - A more compact BoundsState object can be created using the PointBounds class
                    
    *Optional*: 
        **inverse_map**: A callable where the first positional argument is for input 'y' values
            - An 'inverse' mapping function `G`[`y`] -> `x` 
            - Used for more efficient bijections, should form a valid bijection with the forward map
                (See *Bijections with Continuous Functions* and *Creating Inverse Functions*)
            
        **`y`-bounds**
            - The minimum and maximum `y` bound values
            - `x` bounds generally determine the `y` bounds
        
        **direction**: 
            - Assuming the forward function describes monotonic function, this attributes indicates whether the function is increasing or decreasing
            - If increasing, a lower/upper bound contains contains a pair of minimum or maximum 'x' and 'y' values
            - If decreasing, a lower/upper bound contains one minimum 'x' or 'y' value, and one maximum 'x' or 'y' value
        
        **id**:
            - The identifer of a RealBijection (see __init__())
        
        
    (see **init()** for details)
    
    Bijections with Continuous Functions
    ---
    
    If memory efficiency is not a concern, then the creating bijections between the real domains `X` and `Y` is usually simple. A valid bijection can be formed by mapping `x` values with `(x,y)` points.
        -  In practice, this means storing and returning both the `x` and `y` values during evolutions, mutations and evaluations 
        -  In this case, only a function for converting `x` values to `y` values is required
            
    If memory efficiency is a concern, then you may be able to decode `y` values using an "inverse" function.
    The mutations and evolutions that use *RealBijection*(s) do not call the inverse function very often, 
    so the time efficiency differences between using vs. not using an inverse function is usually insignificant. 
                
    If the functions `F` and `G` are continuous, this generally implies that `F` and `G` are **strictly monotonic** – i.e always increasing or always decreasing.
        - Most continuous functions `y = f(x)` are not strictly monotonic over all possible `x` (although many common and useful functions are).
        - Even if a `y = f(x)` function is not strictly monotonic, one-to-one mapping between 'x' and 'y' values can often still be created.
            - See *Creating Inverse Functions*
        
    
    Creating Inverse Functions
    ----
    
    The simplest case for mapping `x` and `y = f(x)` values with an inverse function is when `f(x)` is **strictly monotonic**
        - `f(x)` strictly increasing or strictly decreasing (if `xi` < `xj`  then  `f(xi)` < `f(xj)` or `f(xi)` > `f(xj)` for all `x`)
        - In this case, `y = f(x)` is invertible to `x = g(y)` 
            (and this inverse can usually be found immediately with online math engines)

    However, it is not always required that `y = f(x)` be strictly monotonic. **Most functions are strictly monotonic over a particular range of `x` or `y` values**.
        - For example, `y = x^2` is not strictly monotonic and does not have a real inverse. However, it is strictly monotonic and has a real inverse when`x` is in [0, inf) or `x` is in (-inf, 0]  
            - (The inverses are `x = sqrt(y)` and `x = -sqrt(y)` respectively)
        - If only interested in mapping of `y = x^2` when `x` >= `0`, then the bijection can be formed immediately with the `x = sqrt(y)` inverse function.
            Otherwise, a bijection can be created for `y = x^2` by combining two *RealBijection* objects and "connecting" them at `x = 0`. 
        - **Most continuous functions `y = f(x)` can theoretically be created by combining some number `f(x)` functions that are strictly monotonic over a particular range**

    Lots of bell curves and unimodal distributions (Normal/Gaussian, Laplace, Cauchy, etc.) can be turned into a `x` <-> `y` mapping, instead of a `x` <-> `(x,y)` mapping. 
        - **Generally, any function that has *convexity* can be described as a memory efficient bijection**; it is only required that an inverse be known/calculable for each "side" of the function. 
        -  Using the previous example, `y = x^2` is *concave upward* function
        - If we have a distribution of y = x^2 values, we can quickly see if a `y` value should map to a positive or negative `x` value.
            - Inpect the y values to the left and/or right of particular y value and see if there is an increase or decrease.
            - If the y value to the right is greater, and/or the value to the left is lesser, then we can usually assume we are on the positive x side of y = x^2)
                Therefore, we'd know to use `x = sqrt(y)` instead of `x = -sqrt(y)`. 
        - Note, the edge cases around the global minimum or maximum value should be accounted for. 
            (the `y` values to the right or left may have been mapped from an x value on the other "side" of the global min/max)

    If you're only dealing with *symmetric* outputs then evolutions and mutations can be made even more efficient by only computing values from one "side".
        - The values from one side can be copied, reversed and appended to the start/end of a output distribution. (see *SymmetricDistributions*)

    The previous examples are probably the most common types of functions to use with an inverse function (as they are usually easy to implement).
    However, those example aren't exhaustive; there are lots of other functions and techniques you could exploit to make more efficient bijections. 
        For example
    
        - **Periodic functions**
            - You only need to know and evolve are parameters like frequency, amplitude, number of cycles, and the starting point within a cycle
                Then loop over a `y = f(x)` function, which is specific to a single cycle
            - The inverse map `x` -> `g(y)` is just logic to map a y value to a specific point in a cycle
            
        - **Strict Bounds**
            - For example, limiting or fixing:
                - the first and last value of outputs
                - the `x` distance between the first and last output 
                - the `x` distance between adjacent outputs
            - This **discretization** of the `y = f(x)` function can produce known relationships between y values, 
            which can be exploited to infer the 'x' value associated with a particular y value of a distribution.

        etc...
    """
   
    def __init__(self, 
                 forward_function: Union[callable, partial], 
                 x_point_bounds: PointBounds | BoundsState,
                 inverse_function: Union[callable, partial] | None = None, 
                 fixed_forward_keywords: dict = {}, 
                 fixed_inverse_keywords: dict = {}, 
                 compute_y_bounds: bool = True, 
                 y_min = None, 
                 y_max = None, 
                 direction: bool | None = None, 
                 raise_execution_errors: bool = True, 
                 unique_id: Union[Hashable, None] = None):
        """ 
        Args:
        
            **forward_function** (Union[callable, partial]): The function mapping `x` values to `y` values (see notes on functions below)
            
            **x_point_bounds** (PointBounds | BoundsState): All bounds for the `x` number line, and the bounds for placing points on that line (see `PountBounds`). Defaults to None
                        - Can also be a `BoundsState` object, which created with `PointBounds.create_bounds_state()`. This is a more compact version of PointBounds, but does not have "setter" methods.
                        
            **inverse_function** (Union[callable, partial]): The function mapping `y` values to `x` values (see notes on functions below). Defaults to None
            
            **fixed_forward_args** (dict, optional): Fixed parameter values for the *forward_function*, excluding than the input `x` value. Defaults to empty dictionary.
                -   If inputted *forward_function* is not a `functools.partial` object, these keyword arguments will be used to create one (if more than one parameter exists in the function)
                -   See example below
                
            **fixed_inverse_args** (dict, optional): Fixed parameter values for the *inverse_function*, excluding than the input `y` value.. Defaults to empty dictionary.
                -   Same as *fixed_forward_args*, but for the inverse function
            
            **compute_y_bounds**: If True, all y_bounds are computed with the forward function. The direction attribute will also be set. Defaults to True.
            
            **y_min** A `y` bound value. Defaults to None.
                - May be an lower or upper bound value depending on the direction of the function(s) (see *direction*)
                - Must be None, or less than or equal to *y_max* 
                - If compute_y_bounds is True, this parameter is ignored
            
            **y_max**: A `y` bound value. Defaults to None.
                - May be an lower or upper bound value depending on the direction of the function (see *direction*)
                - Must be None, or greater than or equal to *y_min*
                - If compute_y_bounds is True, this parameter is ignored
            
            **direction**: Defaults to None.
                - If False (or 0) indicates that the foward / inverse function is *increasing*
                    - The lower bound is `(xmin, ymin)`  and the upper bound is `(xmax, ymax)`
                - If True (or 1), indicates that the forward / inverse function is *decreasing*
                    - The lower bound is `(xmin, ymax)` and the upper bound is `(xmax, ymin)`
                - If None, indicates the direction of the forward / inverse function is unknown, or that the function(s) are not decreasing or increasing.
                    - It will be assumed that the lower and upper bound  is `(xmin, ymin)` and `(xmax, ymax)` (like an increasing function)
                - If compute_y_bounds is True, this parameter is ignored

            **raise_execution_errors**: Whether an error is raised or None is returned when an error occurs calling `forward_map(x)` or `inverse_map(y)`. Defaults to True.
                - If True, the error is raised. If False, None is returned. 
                
            **unique_id** (Any, Union[Hashable, None]): The identifer of this bijection.
                - If one is not provided, a `uuid4` object will be used (see `uuid` docs and notes below)
                - This allows `RealBijection` objects to be used in sets, dictionaries, etc.

        Important Notes
        -----
        **Note on Mapping Functions**
            - The `x` and `y` input values *must* be the first positional argument of the forward and inverse functions
                - (See example below)
            - Unless you plan to update the keyword parameters later, all parameters without default values (other than the first positional argument), 
            should be set in the `functools.partial` objects and/or provided by *fixed_forward_args* and *fixed_inverse_args*. 
            -  Since the first positional argument (`x` and `y`) of a function should not be provided, this implies that a `functools.partial` object should not have any positional arguments set.
            - A "fixed" map just means that the required keyword parameters' values are known, but any keyword parameter value can be updated at any time.
            
        **Note on Multi-Processing**
            - If you plan to use a `RealBijection` in a multiprocessing algorithm (and/or as shared memory), the mappings functions be should defined globally (i.e not returned or defined inside another function or class). 
            More generally, mappings functions and their argument types must be pickleable. If both the *unique_id* and mappings functions are pickleable, then `RealBijection` objects are also pickleable. 
                - (See example below)
            - If providing a *unique_id*, and you plan to create these *RealBijection* objects during a process, ensure that the *unique_id* is safe for multiprocessing / multithreading. It is also important to ensure that these ids are actually unique.
            - If a *unique_id* is not provided, `uuid4` is used as default. **Although *uuid4* is usually multiprocessing-safe, this is not guaranteed and may depend on your operating system**. 
            You can the check the multiprocessing safety of a `uuid` object with `uuid.SafeUUID()` (see the `uuid` docs).
            - Note that, this class currently doesn't provide any guarantees on the synchronization of 'set' operations. If updating bounds or parameters during multiprocessing, synchronization logic will have to implemented elsewhere. 
            For mutations and evolutions, these operations generally happen *between* the multi-processed evaluations; in this case, synchronization of set operations is not required.
        
        An Example
        ------
        
                    def linearForward(x, m, b):
                        return (m * x) + b

                    def linearInverse(y, m, b):
                        return (y - b) / m

                    def simple_linear_bijection(m, b):
                        assert m != 0
                    
                        return (
                        
                            functools.partial(linearForward, m = m, b = b), 
                        
                            functools.partial(linearInverse, m = m, b = b)
                        )
                 
        The returned *functools.partial* objects of *simple_linear_bijection()* can be inputted into the initialization function and the "forward" and "inverse" functions. 
        
        Equivalently, the *linearForward()* and *linearInverse()* functions can be inputted directly, 
        and dictionaries with `m` and `b` key-value pairs could be provided as the *fixed_forward_args* and *fixed_inverse_args* parameters.
            (Alternatively, they could be provided at a later time with the *update_keyword_args()* method)
            
        If a lower x and upper x bound are provided, then the y bounds could be computed automatically with *compute_missing_bounds*. 
        If y_min and y_max are provided, then the x bounds could be computed with *compute_missing_bounds*.
        
        If m > 0, the direction should be set to `False` (or `True` if m < 0). The direction could also be set automatically with *compute_missing_bounds*
            
        """  
        self._id = None 
        if unique_id is not None:
            if not isinstance(unique_id, Hashable):
                raise ValueError("The unique id is not a hashable.")
            self._id = unique_id
        else:
            self._id = uuid.uuid4()
        
        self._forward_map = None
        self._inverse_map = None
        
        # Inspect / Validate forward function and parameters
        executable_forward = False
        required_not_provided_forward = None
        try:
            self._forward_map, required_not_provided_forward = self._validate_map_func(forward_function, fixed_forward_keywords)
            if not required_not_provided_forward:
                executable_forward = True
        except AssertionError as e:
            print(f"Invalid forward function: ")
            raise 
        except KeyError as e:
            print(f"Invalid parameters provided for forward function: ")
            raise
        except:
            raise
        
        # Inspect / Validate inverse function and parameters
        if inverse_function is not None:
            try:
                self._inverse_map, _ = self._validate_map_func(inverse_function, fixed_inverse_keywords)        
            except AssertionError as e:
                print(f"Invalid inverse function: ")
                raise 
            except KeyError as e:
                print(f"Invalid parameters provided for inverse function: ")
                raise
            except:
                raise
        
        if not isinstance(x_point_bounds, (PointBounds, BoundsState)):
            raise ValueError(f"'x_point_bounds' must be a PointBounds or BoundsState object, got {type(x_point_bounds)}. ")
        self._x_point_bounds = x_point_bounds
        self._return_type = x_point_bounds.dtype
        self._raise_error = raise_execution_errors
        self._decreasing = None if direction is None or compute_y_bounds else bool(direction)
        self._y_bounds = None
        
        # Compute missing y bounds
        self._left_y_bounds = None
        self._right_y_bounds = None
        if compute_y_bounds and executable_forward:
            self.set_left_right_y_bounds()
        else:
            if y_min is not None and y_max is not None and not (y_min <= y_max):
                raise ValueError("y_min must be less than or equal to y_max")
            self._y_bounds = (y_min, y_max)

   
    @staticmethod
    def _validate_map_func(map_func: Union[partial, callable], fixed_parameters: dict):
        """
        Find the count of all required parameters and all parameters w/o default values, excluding first positional argument
        
        Returns (tuple):
            map_func in correct format (callable), names of required parameters not provided (set) 
        """      
        assert(isinstance(map_func, partial) or callable(map_func)) , "Map function input is not a callable (or functools.partial) object"
        if not isinstance(fixed_parameters, dict):
            warn("fixed_parameters is not a dictionary")
            fixed_parameters = {}
        
        sig = None
        if isinstance(map_func, partial):
            sig = inspect.signature(map_func.func)
        else:
            sig = inspect.signature(map_func)
            
        param_names = list(sig.parameters.keys())
        num_params = len(param_names)
        assert(num_params > 0), "No function parameters exist."
        
        input_arg_name = param_names[0]
        if len(param_names) == 1:
            if isinstance(map_func, partial): # remove function from partial
                return map_func.func, set()
            return map_func, set()

        param_vals = list(sig.parameters.values())
        remaining_names = param_names[1:]
        remaining_vals = param_vals[1:]
        
        for provided in fixed_parameters.keys():
            if provided not in remaining_names:
                raise KeyError(f"Invalid fixed parameter name {provided}")
            elif provided == input_arg_name:
                raise KeyError(f"Invalid fixed parameter name {provided}. The first position argument cannot be fixed.")
                
        if isinstance(map_func, partial) and map_func.args != ():
            KeyError(f"Positional arguments cannot set in the input 'partial' function, only keyword arguments.")
        
        required_not_provided = set()
        for pname, pval in zip(remaining_names, remaining_vals):
            if pval.default is inspect.Parameter.empty:
                # User provided a non-default parameter in fixed param dictionary or in the partial
                if pname in fixed_parameters or (isinstance(map_func, partial) and pname in map_func.keywords.keys()): 
                    continue
                # A non-default param is not provided 
                else: 
                    required_not_provided.add(pname)
                    
        if not isinstance(map_func,  partial):
            partial_obj = partial(map_func, **fixed_parameters)
            return partial_obj, required_not_provided
        else:
            map_func.keywords.update(fixed_parameters)
            return map_func, required_not_provided
      
    def update_keyword_args(self, forward_args: dict | None = None, inverse_args: dict | None  = None):
        "Update the keyword argument values of the forward mapping function and/or inverse mapping function"
        if forward_args:
            if not isinstance(self.forward_function, partial):
                warn("The forward mapping function does not contain keyword arguments. No update occured")
                return
            assert(isinstance(forward_args, dict)), "forward_args is not a dictionary"
            try:
                self._forward_map.keywords.update(forward_args)
            except Exception as e:
                warn(f"Unable to update keyword arguments of the forward map: {e}")
        
            
        if inverse_args:
            if not self._inverse_map:
                warn("The inverse mapping function is not set. Keywords cannot be updated")
                return
            if not isinstance(self.forward_function, partial):
                warn("The inverse mapping function does not contain keyword arguments. No update occured")
                return
            assert(isinstance(inverse_args, dict)), "inverse_args is not a dictionary"
            try:
                self._inverse_map.keywords.update(inverse_args)
            except Exception as e:
                warn(f"Unable to update keyword arguments of the inverse map: {e}")
                
    def set_direction(self, direction):
        """
        `
        Set the monotonic direction of the forward function / bijection. 
        
        direction = 0, or False, indicates the function is **increasing**.
        
        direction = 1, or True, indicates the function is **decreasing**
        
        None indicates the direction is unknown.`
        """        
        if direction is None:
            self._decreasing = None
        else:
            self._decreasing = bool(direction)

    def set_raise_execution_errors(self, raise_execution_errors: bool):
        self._raise_error = raise_execution_errors
    
    def set_left_right_y_bounds(self):
        """ 
        - Set *left* y bounds (mapped from the first point x bounds) and the *right* y bounds (mapped from the last point x bounds).
        Unlike the y-bounds attribute, these attributes will not always be sorted in ascending order - rather they are sorted depending on the direction attribute
        
        If y_min or y_max is not set (the y_bounds), then there will be an attempt to set these values first. The direction will also be set.
        
        If both y_bounds are set, then the direction should **already be set**.
       """
       
        # compute y min and y_max if they do not exist
        if self._y_bounds is None or self._y_bounds[0] is None or self._y_bounds[1] is None:
            try:
                compute_and_set_missing_y_bounds(self, set_direction=True, raise_execution_errors=True)
            except Exception as e:
                if self._raise_error:
                    print("Unable to set missing y bounds")
                    raise
                else:
                    warn(f"Unable to set missing y bounds {e}")
                    
        y_min, y_max = self._y_bounds
        y_from_max_first = None
        y_from_min_last = None
        try:
            y_from_max_first = self.fixed_forward_map(self.point_bounds.max_first_point)
        except:
            warn("Unable to set y bound associated with the first x upper bound")
        else:
            if y_from_max_first is None:
                warn("Unable to set y bound associated with the first x upper bound")
                
        try:
            y_from_min_last = self.fixed_forward_map(self.point_bounds.min_last_point)
        except:
            warn("Unable to set y bound associated with the last x lower bound")
        else:
            if y_from_min_last is None:
                warn("Unable to set y bound associated with the first x upper bound")
  
        # If increase right = y_min then y_from_max_first, If decrease then y_max then y_from_max_first
        self._left_y_bounds = (y_min, y_from_max_first) if not self._decreasing else (y_max, y_from_max_first)
        self._right_y_bounds = (y_from_min_last, y_max) if not self._decreasing else (y_from_min_last, y_min) 
        # print(f"y_min: {y_min}, y_from_max_first: {y_from_max_first}, y_from_min_last {y_from_min_last}, y_max {y_max}")

    def fixed_forward_map(self,x):
        """
        Execute the forward map function with an input 'x' value
        
        Note: All required parameters of the forward function (excluding the first positional argument), should have already been provided 
            See `variable_forward_map()`
        """        
        try:
            return self.dtype(self._forward_map(x))
        except Exception:
            if self._raise_error:
                print(f"Error with execution of forward map function where x = {x}: ")
                raise
            else:
                return None
    
    def fixed_inverse_map(self,y):
        """
        Execute the forward map function with an input 'x' value
        
        Note: The inverse function should exist, and all required parameters of the inverse function (excluding the first positional argument), should have already been provided.
            See `variable_inverse_map()`
        """    
        assert(self._inverse_map is not None), "The inverse map function has not been set"
        try:
            return self.dtype(self._inverse_map(y))
        except Exception:
            if self._raise_error:
                print(f"Error with execution of inverse map function where y = {y}: ")
                raise
            else:
                return None
    
    def variable_forward_map(self, x, **kwargs):
        """`
        Execute the forward map with an 'x' value and new *kwargs* keyword parameters (if the forward mapping function has more than 1 argument)
        
            - Note, the *kwargs* values will override the parameters set in the functools partial object, but will not update those parameter values permanently
                (see *update_keywords_args()*)
        """        
        if not isinstance(self.forward_function, partial) or not kwargs:
            return self.fixed_forward_map(x)

        try:
            temp_args = self._forward_map.keywords | kwargs 
            return self.dtype(self.forward_function.func(x, **temp_args))
        except Exception:
            if self._raise_error:
                print(f"Error with execution of forward map function where x = {x}: ")
                raise
            else:
                return None
    
    def variable_inverse_map(self, y, **kwargs):
        """`
        Execute the inverse map with an 'y' value and new *kwargs* keyword parameters (if the inverse mapping function has more than 1 argument)
        
            - Note, the *kwargs* values will override the keyword parameters set in the functools partial object, but will not update those parameters values permanently
                (see *update_keywords_args()*)
        """
        if not isinstance(self.inverse_function, partial) or not kwargs:
            return self.fixed_inverse_map(y)
        try:
            temp_args = self._inverse_map.keywords | kwargs 
            return self._return_type(self._inverse_map.func(y, **temp_args))
        except Exception:
            if self._raise_error:
                print(f"Error with execution of inverse map function where y = {y}: ")
                raise
            else:
                return None
            
    def create_distribution(self, start_x, num_points, separation, reverse_x = False):
        """Create a distribution of "y" values 

        Assumes that all parameters of the forward function are set

        Will raise errors regardless of raise_error attribute
        
        Args:
            start_x (numeric): The first x value to map
            num_points (numeric): Number of y values in output distribution
            separation (numeric): The fixed separation between x values
            reverse_x (bool): The order of x values when mapping to y values. Defaults to False
                If True, output y values are mapped from last x to start x
                
                if False, output y values are mapped from start x to last x
        """        
        x_vals = (start_x + separation*i for i in range(num_points)) 
        iter = map(self.forward_function, x_vals)
        try:
            distribution = np.fromiter(iter=iter, dtype = self.dtype, count = num_points)
            if reverse_x:
                distribution = distribution[::-1]
            return distribution
        except Exception as e:
            print(f"Error creating output with x0 = {start_x}, num_points = {num_points}, separation = {separation}: \n {e}")
            raise
   
    def vectorized_create_distribution(self, start_x, num_points, separation, reverse_x = False):
        """Assuming the forward map can accept numpy arrays as input, a faster implementation to create output distributions
            
            Args:
                start_x (numeric): The first x value to map
                num_points (numeric): Number of y values in output distribution
                separation (numeric): The fixed separation between x values
                reverse_x (bool): The order of x values when mapping to y values. Defaults to False
                    If True, output y values are mapped from last x to start x
                    
                    if False, output y values are mapped from start x to last x
            """
        i = np.arange(num_points)
        xs = start_x + separation * i 
        try:          
            ys = self.forward_function(xs)                     
            distribution = ys.astype(self._return_type)
            if reverse_x:
                distribution = distribution[::-1]
        except Exception as e:
            print(f"Error creating output with x0 = {start_x}, num_points = {num_points}, separation = {separation}: \n {e}")
            raise
        return distribution
    
    @property
    def id(self) -> Any:
        """Get the unique id of this bijection object"""
        return self.id
    
    @property
    def dtype(self) -> type:
        """Get the return type of the forward/inverse functions and the bound values"""
        return self._return_type
    
    @property
    def forward_function(self) -> Union[callable, partial]:
        """Get the forward mapping function"""
        return self._forward_map
    
    @property
    def inverse_function(self) -> Union[callable, partial, None]:
        """Get the inverse mapping function"""
        return self._inverse_map

    @property
    def point_bounds(self) -> PointBounds | BoundsState:
        """Get the `PointBounds` (or `BoundsState`) object. This object contains all getter methods for 'x'-value, cardinality and separation bounds.
        
        If 'x_point_bounds' is not a BoundsState, then this object should be used to change any `x` value bounds or point bounds"""
        
        return self._x_point_bounds
    
    @property
    def direction(self):
        """Get the direction of the map function or a default value if None
        - Returns True if decreasing, False if increasing, and None if not set / unknown"""
        return self._decreasing
    
    @property
    def y_bounds(self) -> Tuple[Number | None, Number | None]:
        """ `(minimum y value, maximum y value)`
            - If the direction is decreasing (i.e True), the minimum y value is associated with the upper x bound, and the maximum y value is associated with the lower x bound,
        """        
        return self._y_bounds
    
    @property
    def left_y_bounds(self) -> tuple | None:
        """Get *left* y bounds -> associated with the first point x_bounds.
        The order of the values depends on the direction (unlike the y_bounds property)
        
        Will return None if not set
        """
        return self._left_y_bounds

    @property
    def right_y_bounds(self) -> tuple | None:
        """Get *right* y bounds -> associated with the last point x bounds.
        The order of the values depends on the direction (unlike the y_bounds property)
        
        Will return None if not set
        """
        return self._right_y_bounds
    
    def x_bound_str(self) -> str:
        out = "("
        out += "-inf, " if self._x_point_bounds.lower_bound is None else f"{self._x_point_bounds.lower_bound}, "
        out += "inf)" if self._x_point_bounds.upper_bound is None else f"{self._x_point_bounds.upper_bound})"
        return out
    
    def y_bound_str(self) -> str:
        out = "("
        out += "-inf, " if self.y_bounds[0] is None else f"{self.y_bounds[0]}, "
        out += "inf)" if self.y_bounds[1] is None else f"{self.y_bounds[1]})"
        return out
    
    def __eq__(self, other):
        if not isinstance(other, RealBijection):
            return False
        else:
            return self.id == other.id 
    
    def __hash__(self):
        return hash(self.id)
    
def compute_and_set_missing_y_bounds(
    bijection: RealBijection, 
    set_direction = True, 
    raise_execution_errors = True,
    ):
    """`
    
    Find and set a `RealBijection` object's missing y-bound values with its x-bound values and forward map"
    
          
    See `RealBijection`

    Args:
        bijection (RealBijection): A RealBijection object
        
        set_direction (bool, optional): If a direction can be determined, set the direction in the RealBijection. Defaults to True.
        
            - Determined by whether a lower x bound and/or upper x bound value mapped to a smaller/greater y value)

            - Can only be determined if computing both y bounds, or computing one y bound and one is already known
               
        raise_execution_errors (bool, optional): Defaults to True.
            If error occurs executing a lower or upper 'x' bound value with the forward map, determines whether a warning or error is raised. 
            In either case, y bounds will not be set if unable to map a known lower or upper bound x value to a y value
    """    

    point_bounds = bijection.point_bounds
    x_min  = point_bounds.lower_bound
    x_max = point_bounds.upper_bound

    prev_raise_execution_errors = bool(bijection._raise_error)
    bijection.set_raise_execution_errors(True) # Temporarily set execution errors to True to catch errors
    
    # Calculate y bounds from x_min and x_max
    y_from_x_min = None
    y_from_x_max = None
    try: 
        y_from_x_min = bijection.dtype(bijection.forward_function(x_min))
    except Exception as e:
        if raise_execution_errors:
            print(f"Error with calculating the y-bound from the lower x-bound")
            raise
        else:
            warn(f"Error with calculating the y-bound from the lower x-bound {e}. \n No y bounds set.")
            return
    try: 
        y_from_x_max = bijection.dtype(bijection.forward_function(x_max))
    except Exception as e:
        if raise_execution_errors:
            print(f"Error with calculating the y-bound from the upper x-bound: ")
            raise
        else:
            warn(f"Error with calculating the y-bound from the upper x-bound: {e} \n No y bounds set.")
            return 
    
    # Restore previous execution error setting
    bijection.set_raise_execution_errors(prev_raise_execution_errors) 
    
    # Set y bounts and direction
    bijection._y_bounds = (min(y_from_x_min, y_from_x_max), max(y_from_x_min, y_from_x_max))
    if set_direction:
        decreasing = y_from_x_min > y_from_x_max if y_from_x_min != y_from_x_max else None
        bijection.set_direction(decreasing)

def compute_and_set_new_x_bounds(
    bijection: RealBijection, 
    set_direction = True, 
    raise_execution_errors = True
    ):
    """`
    
    Find and set a `RealBijection` object's missing x-bound values with its y-bound values and inverse map"
    
    Will do nothing if not all y-bounds are set
    
    Will raise an error if the computed/known lower x bound an upper x bound values are equal
    
    Note:
        - If both y-bound values are known, and one x-bound value is already set, this function will override that x-bound value
            
        - **If both x bounds values will be known after the computing missing values, it is highly recommended that "set_direction" be True**
  
            - Setting the direction allows you to know which y-bound corresponds to the lower x bound and which corresponds to the upper x bound
                - (For decreasing functions, the greater y-bound value corresponds with the lower x bound, and vice versa)
                
    See `RealBijection`

    Args:
        bijection (RealBijection): A RealBijection object
        
        set_direction (bool, optional): If a direction can be determined, set the direction in the RealBijection. Defaults to True.
        
            - Determined by whether a smaller y bound and/or greater y bound value mapped to a smaller/greater x value)

            - Can only be determined if computing both x bounds, or computing one x bound and one is already known
            
        raise_execution_errors (bool, optional): Defaults to True.
            If error occurs executing a lower or upper 'y' bound value with the inverse map, determines whether a warning or error is raised. 
            In either case, x bounds will not be set if unable to map a known lower or upper bound y value to a x value
    """
    if bijection.inverse_function is None:
        warn("No inverse mapping function has been set. Cannot compute x bounds from y bounds")
        return
    
    y_min, y_max = bijection.y_bounds
    if y_min is None or y_max is None:
        return
    
    prev_raise_execution_errors = bool(bijection._raise_error)
    bijection.set_raise_execution_errors(True) # Temporarily set execution errors to True to catch errors
    
    x_from_y_min = None
    x_from_y_max = None
    try: 
        x_from_y_min = bijection.dtype(bijection.inverse_function(y_min))
    except Exception as e:
        if raise_execution_errors:
            print(f"Error with calculating the x-bound from y-min")
            raise 
        else:
            warn(f"Error with calculating the x-bound from the y-min {e}. \n No x bounds set.")
            return
            
    try: 
        x_from_y_max = bijection.dtype(bijection.inverse_function(y_max))
    except Exception as e:
        if raise_execution_errors:
            print(f"Error with calculating the y-bound from the upper x-bound: ")
            raise 
        else:
            warn(f"Error with calculating the y-bound from the upper x-bound: {e} \n No x bounds set.")
            return 
        
    # Restore previous execution error setting
    bijection.set_raise_execution_errors(prev_raise_execution_errors) 
    if x_from_y_min == x_from_y_max:
        warn(f"Both x bounds are the same value {x_from_y_min}. Cannot set same valued x bounds")
        return
    
    # Set x bounds and direction
    new_x_bounds = (min(x_from_y_min, x_from_y_max), max(x_from_y_min, x_from_y_max))
    if isinstance(bijection.point_bounds, PointBounds):
        bijection.point_bounds.set_lower_bound(new_x_bounds[0])
        bijection.point_bounds.set_upper_bound(new_x_bounds[1])
    elif isinstance(bijection.point_bounds, BoundsState):
        bijection.point_bounds._lower_bound = new_x_bounds[0]
        bijection.point_bounds._upper_bound = new_x_bounds[1]
        _cascade_from_global(bijection.point_bounds)
    
    if set_direction:
        decreasing = x_from_y_min > x_from_y_max
        bijection.set_direction(decreasing)