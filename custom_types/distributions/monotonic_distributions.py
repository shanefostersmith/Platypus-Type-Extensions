# Future work / areas for exploration: 
    # - Batch GPU computing of output distributions (CUDA and Metal Shading)
    # - Decorators/context managers for mappings that can handle vectorized operations
    # -  More sophisticated caching of known x -> y mappings for partial arrays
    # - Optimizing the parameters of a map as well (probably a separate class)
    # - Multi-dimensional distributions (direction bounds instead of point bounds, covariance matrices, etc.)
    # - Gradual tightening of PointBounds over generations
    # - Other potential optimizations 
        # Using elements known numeric optimizations for very large arrays, parameter optimization, or semi-guidance (recontextualized for EP and GP objectives)
            #  Simulated annealing (SciPy dual-annealing, simanneal), BFGS, shared/continually updated 'velocity' term, greater focus of distribution shapes and spreads, etc.
            #  May require a new GlobalEvolution with shared/persistant memory (tracking "directions" that are promising, while keeping novelty of EP and GP searches)

import numpy as np
from collections.abc import Iterable
from ..core import CustomType, LocalMutator, LocalVariator
from .real_bijection import RealBijection
from .point_bounds import PointBounds, bound_tools
from ._distribution_tools import DistributionInfo
from ._mutation_tools import *
from ._crossover_tools import *
from ..real_methods.numba_pcx import normalized_2d_pcx
from ..integer_methods.integer_methods import int_mutation, single_binary_swap
from ..utils import _min_max_norm_convert, _nbits_encode, int_to_gray_encoding, gray_encoding_to_int

class MonotonicDistributions(CustomType):
    """ 
    **Evolve and mutate a discrete set of monotonic distributions in the form of 1-D numpy arrays.**
    
    A distribution (an array of `y` values) is created by mapping a set of evenly-spaced, ordered `x` values to `y` values with a user-defined monotonic function.
    
    Only strictly monotonic distributions are supported. 
    Examples of strictly monotonic functions include:
    -  Exponential functions 
    -  Decay functions
    -  Half-Life functions
    -  Linear functions where the slope != 0
    -  Quadratic functions where x in range (0, inf)
    -  Logarithmic functions where x in range (0, inf)
    
    See *RealBijection*.
    
    Return Variable
    ------
    - The decoded variable type is a tuple: `(np.ndarray, int)`
    
    - The first element is the distribution: an 1D array of sorted `y` values
        - User may choose if distribution values are sorted in ascending or descending order.
    
    - The second element is the index of the mapping used to create the distribution (index of attribute *map_suite*).
        - For decoding/encoding
        - Used to inspect the function converged upon post-optimization 
        
    Note on *RealBijection* Inputs
    ------
    User inputs an iterable of *RealBijection* objects, where **the inverse functions are required to be set**.
    
    This class assumes that the *direction* attribute of the *RealBijection* objects are correct. 
        - For example, if the *RealBijection* object has no direction set or a direction set to *increasing*, but the function is actually *decreasing*, this may result in unexpected behavior.
    
    It is generally recommended that each *RealBijection* have the same `dtype`, but this is not required.
        - Output arrays will be the same type as the RealBijections' `dtype`
     
    See `RealBijection` and `PointBounds`
     
    Note on Convergence and Use Case:
    -----
    
    This class's primary purpose is to to evolve sets of points using a discrete set of *candidate distibutions* (ie. the bijection maps).
        - Using a discrete set of fixed maps greatly reduces the search space
        - The candidate distribution should generally have a similar shape or rate of change
            - For example, you may have a set of decay functions that you suspect work well for your optimization problem (through domain knowledge / prior optimizations
        - It is more important that the domain/range of the `y` values are similar rather than the domain/range of the `x` values
            - The evolution/mutation methods generally deal with `x` values and bounds in relative terms, while it deals with output `y` values in absolute terms.
            - Because all outputs are sorted in the same order, it is also not particularly important if some maps are increasing functions and others are decreasing, 
            so long as the y bounds and shape are similar in some direction. 
        
    For faster convergence, it is also recommended that somewhat strict bounds are applied to those mappings. 
        - These bounds may start and/or end in the `x` or `y` domain, the number of points in the output distributions, etc (See *PointBounds* for more details)
        - These choices will depend on the type of evolution/mutate methods used and desired behavior of the search 
    
    If the "candidate" distributions have some sort of ordinality, you may input the *RealBijection* objects can be inputted in that order and set the *ordinal_maps attribute. 
        - Methods that evolve/mutate the `x` -> `y` maps interpret the maps' indices as meaningful (and will use integer methods, rather than random choice, to evolve/mutate maps). 
        - This type of behavior may not be desired if the goal is more of a novelty search
    """
    def __init__(
        self, 
        mappings: Iterable[RealBijection],
        local_variator = None,
        local_mutator = None,
        ordinal_maps = False,
        max_points: int = 1000,  
        sort_ascending = True,
        use_cache = False,
        max_cache_size = 25):
        """
        Args:
            mappings (Iterable[RealBijection]): An iterable of *RealBijection* objects with the inverse function set (see RealBijection)
            local_variator (LocalVariator, optional): Cannot be a LocalMutator, should have MonotonicDistributions registered in _supported_types. Defaults to None.
            local_mutator (LocalMutator, optional): A LocalMutator, should have MonotonicDistributions registered in _supported_types. Defaults to None.
            ordinal_maps (bool, optional): Indicates that the RealBijection objects are inputted in some sort of order. Defaults to False.
            max_points (int, optional): If a *RealBijection*'s 'max_points' bound not already set, this default value will be used. Defaults to 1000.
            sort_ascending (bool, optional): Indictes whether to return distributions in ascending or descending order. Defaults to True.
            use_cache (bool, optional): Cache mappings from encodings -> decoding. Defaults to False.
                - If True, the decode method is wrapped in a fixed-sized LRU cache. If an encoding is in the cache, then creating the output distribution can be skipped
                - Using a cache can result in signficant spped ups, particularly if the mutation probabilties are low and/or there are a small number of candidate distributions
            max_cache_size (int, optional): If `use_cache` is True, then this value indicates the maximum number of encoding -> decoding mappings that can be stored at once. Defaults to 25.
                -  As a general rule of thumb, the cache size does need to be much larger than the 'offspring size' of the Algorithm.
                Or, for algorithms like NSGAII, around 0.5x - 1x the size of the population. 
                - This choice will depend on computation vs. memory efficiency requirements and probability gate values.

        Raises:
            ValueError: If the inverse function of any RealBijection map is not set

        """        
        self.map_suite = []
        self.num_functions = 0
        self.global_min_points = np.inf
        self.global_max_points = -np.inf
        self.global_min_y = np.inf
        self.global_max_y = -np.inf
        max_points = int(max_points)
        self.global_return_type = None
        
        # Create/validate suite of mappings
        for i, bijection in enumerate(mappings): 
            assert isinstance(bijection, RealBijection), f"Element {i} is not a RealBijection object."
            x_bounds = bijection.point_bounds
            if bijection.inverse_function is None:
                raise ValueError(f"RealBijection {i} does not have an inverse function set")
            bijection.set_raise_execution_errors(True)

            # Check num points
            curr_min_points = x_bounds.min_points
            if curr_min_points < self.global_min_points:
                self.global_min_points = curr_min_points
                
            if np.isinf(x_bounds.max_points): # Set scalar max points
                if isinstance(x_bounds, PointBounds):
                    x_bounds.set_max_points(max(x_bounds.min_points, max_points))
                else:
                    self._add_max_points(x_bounds, max_points)
                    
            curr_max_points = x_bounds.max_points
            if curr_max_points > self.global_max_points:
                self.global_max_points = curr_max_points
           
            # Ensure y-bounds are known for y-based crossovers
            left_y, second_y = bijection.left_y_bounds
            penult_y, right_y = bijection.right_y_bounds
            if any(var is None for var in (left_y, second_y, penult_y, right_y)):
                bijection.set_left_right_y_bounds()
                left_y, second_y = bijection.left_y_bounds
                penult_y, right_y = bijection.right_y_bounds
                assert all(var is not None for var in (left_y, second_y, penult_y, right_y)), "Error occured, all y bounds should have been set" 
                
            if self.global_min_y > bijection.y_bounds[0]:
                self.global_min_y = bijection.y_bounds[0]
            if self.global_max_y < bijection.y_bounds[1]:
                self.global_max_y = bijection.y_bounds[1]
            self.map_suite.append(bijection)
            self.num_functions += 1
     
        if self.num_functions == 0:
            ValueError("Must provide at least one RealBijection object")

        self.sort_ascending = sort_ascending
        self.ordinal_maps = ordinal_maps
        encoding_memoization_type = 'cache' if use_cache else None
        if use_cache and use_cache == 'single':
            encoding_memoization_type = 'single' # experimental
        super().__init__(
            local_variator=local_variator, 
            local_mutator=local_mutator,
            encoding_memoization_type=encoding_memoization_type,
            max_cache_size=max_cache_size)
    
    def _add_max_points(self, bounds: bound_tools.BoundsState, default_max_points: int):
        """Adds default max points when using a BoundsState"""
        default_max_points = max(bounds.min_points, default_max_points)
        bound_tools._cascade_from_points(default_max_points, from_max_points=True)
    
    def rand(self):
        
        rand_function_idx = 0 if self.num_functions == 1 else np.random.randint(self.num_functions)
        bijection: RealBijection = self.map_suite[rand_function_idx] 
        x_bounds = bijection.point_bounds
        
        x_min, max_first_x = x_bounds.first_point_bounds
        min_last_x, x_max = x_bounds.last_point_bounds
 
        # Get start of random output
        output_start = None
        output_width = x_bounds.fixed_width
        true_min_width = x_bounds.true_min_width
        true_max_width = x_bounds.true_max_width
        output_start = x_min if x_min == max_first_x else np.random.uniform(x_min, max_first_x)

        # Get a random width of output
        if output_width is None:
            curr_max_width = min(x_max - output_start, true_max_width)
            curr_min_width = true_min_width
            if output_start < min_last_x:
                curr_min_width = max(min_last_x - output_start, true_min_width)
            output_width = curr_min_width if curr_min_width == curr_max_width else np.random.uniform(curr_min_width, curr_max_width)
        
        # Get random points
        true_min_points, true_max_points = x_bounds.get_conditional_cardinality_with_width(output_width)
        output_points = true_min_points if true_min_points == true_max_points else np.random.randint(true_min_points, true_max_points+1)
        return DistributionInfo(
            map_index=rand_function_idx,
            num_points= output_points,
            separation= output_width / x_bounds.dtype(output_points - 1),
            output_min_x = output_start,
            output_max_x = output_start + output_width
        )
    
    def encode(self, value: tuple):
        y_distribution, map_idx = value
        bijection: RealBijection = self.map_suite[map_idx]
        x_bounds = bijection.point_bounds
        
        num_points = len(y_distribution)
        assert num_points >= x_bounds.min_points, f"Too few points in current distribution ({num_points} in distribution, minimum is {x_bounds.min_points})"
        
        first_idx_x = bijection.fixed_inverse_map(y_distribution[0])
        last_idx_x = bijection.fixed_inverse_map(y_distribution[-1]) 
        assert not (first_idx_x is None or last_idx_x is None), "The inverse function returned None values"
        assert first_idx_x != last_idx_x, "The first 'x' value in current distribution equals the last 'x' value"
        
        output_min_x = min(first_idx_x, last_idx_x)
        output_max_x = max(first_idx_x, last_idx_x)
        separation = (output_max_x - output_min_x) / bijection._return_type(num_points - 1)
        
        distribution_info = DistributionInfo(
            map_index = map_idx,
            num_points = num_points,
            separation = separation,
            output_min_x = output_min_x,
            output_max_x = output_max_x)
            
        return distribution_info
    
    def decode(self, value: DistributionInfo):
        bijection: RealBijection = self.map_suite[value.map_index]
        # Create distibution
        direction = False if bijection.direction is None else bijection.direction
        reverse = direction == self.sort_ascending
        try:
            distribution = bijection.create_distribution(value.output_min_x, value.num_points, value.separation, reverse)
            return (distribution, value.map_index)
        except:
            print(f"Error with bijection {value.map_index}: ")
            raise
        
    def __str__(self):
        return f"Monotonic Sets: ({self.num_functions})"

class FixedMapConversion(LocalMutator):
    """A LocalMutator for MonotonicDistributios (and subclasses)
    
    Applies relative x values of a previous solution to new distribution
    
    If a MonotonicDistributions type has the *ordinal_maps* attribute set to True,
    then a bit flip mutation is used. Otherwise, offspring maps are chosen randomly from the candidate distributions.
    
    """

    _supported_types = (MonotonicDistributions)
    __slots__ = ("map_conversion_probability", "y_based_conversion")
    
    def __init__(self, map_conversion_probability = 0.1, y_based_conversion = False):
        """
        Args:
            map_conversion_probability (float, optional): The probability an offspring's map will be changed. Defaults to 0.1.
            y_based_conversion (bool, optional): Indicates which domain to prioritize when converting output values to the new map. Defaults to False.
            - If True, keep the output start and end 'y' values as close as possible to the previous 'y' values during the conversion.
            - If False, keep the relative start and end 'x' values as close as possible to the previous relative 'x' values during the conversion
                -- "relative" = normalized to the width of the 'x' range `(x_value / (x_max - x_min)`)
        """        
        self.map_conversion_probability = map_conversion_probability
        self.y_based_conversion = y_based_conversion
        
    def mutate(self, custom_type: MonotonicDistributions, offspring_solution, variable_index, **kwargs):
        mapping_mutation_prob = 0 if custom_type.num_functions == 1 else self.map_conversion_probability
        if not custom_type.do_mutation or not (mapping_mutation_prob > 0 and np.random.uniform() < mapping_mutation_prob):
            return

        y_based = self.y_based_conversion
        distribution_info: DistributionInfo = offspring_solution.variables[variable_index]

        new_map_idx = None
        if custom_type.num_functions == 2:
            new_map_idx = ~distribution_info.map_index
        elif custom_type.ordinal_maps: #TODO change to gray
            binary_map_idx = int_to_gray_encoding(distribution_info.map_index, 0, custom_type.num_functions - 1)
            new_map_bits, mutated = int_mutation(binary_map_idx, 0.0)
            if not mutated:
                return
            new_map_idx = gray_encoding_to_int(0, custom_type.num_functions - 1, new_map_bits)
        else:
            new_map_idx = np.random.randint(custom_type.num_functions)
        
        if new_map_idx == distribution_info.map_index:
            return
        
        self._execute_map_conversion(custom_type=custom_type, distribution_info = distribution_info, 
                                     previous_map = distribution_info.map_index, new_map = new_map_idx, 
                                     y_based=y_based)
        
        offspring_solution.evaluated = False
    
    @staticmethod            
    def _execute_map_conversion(custom_type: MonotonicDistributions, distribution_info: DistributionInfo, previous_map, new_map, y_based):
        new_bijection: RealBijection = custom_type.map_suite[new_map]
        prev_bijection: RealBijection = custom_type.map_suite[previous_map]
        new_min_x = None
        new_max_x = None
        new_separation = None
        new_num_points = None
        if y_based:
            y1 = prev_bijection.fixed_forward_map(distribution_info.output_min_x)
            y2 = prev_bijection.fixed_forward_map(distribution_info.output_max_x)
            new_min_x, new_max_x, new_separation, new_num_points = map_conversion_y_based(distribution_info, prev_output_min_y = min(y1,y2), prev_output_max_y = max(y1,y2), new_bijection = new_bijection)
            
        else:
            new_min_x, new_max_x, new_separation, new_num_points = map_conversion_x_based(distribution_info, prev_bijection, new_bijection)
        
        distribution_info.map_index = new_map
        distribution_info.output_min_x = new_min_x
        distribution_info.output_max_x = new_max_x
        distribution_info.separation = new_separation
        distribution_info.num_points = new_num_points
        
class DistributionShift(LocalMutator):
    """Shifts a distribution along the x-axis
    
    No change in number of points or separation between points.
    
    If a RealBijection/PointBounds object defines a "fixed" first or last point (the lower bound is equal to first bounds upper bound, or vice versa),
    then no shift can occur"""
    _supported_types = MonotonicDistributions
    __slots__ = ("shift_probability", "shift_alpha", "shift_beta")
    
    def __init__(self, shift_probability = 0.5, shift_alpha = 1.5, shift_beta = 6):
        """Note, shift_alpha and shift_beta parameterize a `numpy.beta(alpha, beta)` distribution

        Args:
            shift_probability (float, optional): _description_. Defaults to 0.5.
            shift_alpha (float, optional): Controls the distribution of shift magnitudes. Defaults to 1.5.
            shift_beta (int, optional): Controls the distribution of shift magnitudes.Defaults to 6.

        Raises:
            ValueError: If shift_alpha <= 0 or shift_beta <= 0
        """        
        if shift_alpha <= 0 or shift_beta <= 0:
            raise ValueError("Both 'shift_alpha' and 'shift_beta' must be greater than 0")
        self.shift_probability = shift_probability
        self.shift_alpha = shift_alpha
        self.shift_beta = shift_beta

    def mutate(self, custom_type: MonotonicDistributions, offspring_solution, variable_index, **kwargs):
        
        distribution_info: DistributionInfo = offspring_solution.variables[variable_index]
        bijection: RealBijection = custom_type.map_suite[distribution_info.map_index]
        x_bounds = bijection.point_bounds
        shift_mutation_prob = 0 if x_bounds.lower_bound == x_bounds.max_first_point or x_bounds.upper_bound == x_bounds.min_last_point else self.shift_probability
        if not custom_type.do_mutation or not (shift_mutation_prob > 0 and np.random.uniform() < shift_mutation_prob):
            return
        
        shift_alpha = self.shift_alpha
        shift_beta = self.shift_alpha
        x_shift = shift_mutation(
            x_bounds, 
            distribution_info.output_min_x, distribution_info.output_max_x, 
            shift_alpha, shift_beta, 
            return_type = bijection.dtype)
        
        if x_shift == 0:
            return
        
        distribution_info.output_min_x += x_shift
        distribution_info.output_max_x += x_shift
        offspring_solution.evaluated = False

class SampleCountMutation(LocalMutator):
    """A Local Mutator for MonotonicDistributions and subclasses

    If a RealBijection defines a "fixed width" bound (or very close to one), then the separation between points changes inversely to the number of to points added / removed.

    Otherwise, some number of points are added / removed from a one side of the distribution. The separation between points does not change.
    
    """
    _supported_types = MonotonicDistributions
    __slots__ = ("mutation_probability", "mutation_count_limit")
    
    def __init__(self, mutation_probability = 0.1, mutation_count_limit: int | None = None):
        """
        Args:
            mutation_probability (float, optional): Defaults to 0.1.
            mutation_count_limit (int | None, optional): An inclusive upper limit on how many points can be added or removed during a mutation. Defaults to None.
                - If None, the limit is only defined by the cardinality bounds of the current map
        """        
        self.mutation_probability = mutation_probability
        self.mutation_count_limit = None
        if mutation_count_limit is not None:
            mutation_count_limit = int(max(1, mutation_count_limit))
    
    def mutate(self, custom_type: MonotonicDistributions, offspring_solution, variable_index, **kwargs):
        distribution_info: DistributionInfo = offspring_solution.variables[variable_index]
        bijection: RealBijection = custom_type.map_suite[distribution_info.map_index]
        x_bounds = bijection.point_bounds
        count_mutation_prob = self.mutation_probability
        
        if not custom_type.do_mutation or x_bounds.min_points == x_bounds.max_points or not (
            count_mutation_prob > 0 and np.random.uniform() < count_mutation_prob):
            return
        
        count_limit = self.mutation_count_limit
        point_difference, diff_at_min_x, separation_change = count_mutation(
            x_bounds, distribution_info.output_min_x, distribution_info.output_max_x,
            distribution_info.separation, distribution_info.num_points, count_limit=count_limit)
        
        if not point_difference:
            return

        distribution_info.num_points += point_difference
        if separation_change:
            distribution_info.separation = separation_change
        else:
            if diff_at_min_x: # Removing points -> start at higher x, Adding points -> start at lower x 
                distribution_info.output_min_x += -1 * point_difference * distribution_info.separation
            else: # Removing points -> end at lower x ,  Adding points -> end at higher x 
                distribution_info.output_max_x += point_difference * distribution_info.separation
                
        offspring_solution.evaluated = False

class PointSeparationMutation(LocalMutator):
    """ A LocalMutator for MonotonicDistributions and subclasses
    
    Mutates the distance between points relative to the `x` values. The number of points in the output distribution does not change.
    
    If a RealBijection/PointBounds object has defined a fixed width, then `SampleCountMutation` should be used instead.
    
    (see SampleCountMutation)
    """
    _supported_types = MonotonicDistributions
    __slots__ = ("mutation_probability", "separation_alpha", "separation_beta")
    
    def __init__(self, mutation_probability = 0.1, separation_alpha = 1, separation_beta = 10):
        """Note, separation_alpha and separation_beta parameterize a `numpy.beta(alpha, beta)` distribution

        Args:
            mutation_probability (float, optional): Defaults to 0.1.
            separation_alpha (int, optional): Controls the distribution of output separation values. Defaults to 1.
            separation_beta (int, optional): Controls the distribution of output separation values. Defaults to 10.
        """        
        self.mutation_probability = mutation_probability
        self.separation_alpha = separation_alpha
        self.separation_beta = separation_beta
    
    def mutate(self, custom_type: MonotonicDistributions, offspring_solution, variable_index, **kwargs):
        distribution_info: DistributionInfo = offspring_solution.variables[variable_index]
        bijection: RealBijection = custom_type.map_suite[distribution_info.map_index]
        x_bounds = bijection.point_bounds
        separation_mutation_prob = self.mutation_probability
        if not custom_type.do_mutation or x_bounds.min_separation == x_bounds.max_separation or not (
            separation_mutation_prob > 0 and np.random.uniform() < separation_mutation_prob):
            return
        
        separation_alpha = self.separation_alpha
        separation_beta = self.separation_beta
        new_out_min, new_out_max, new_separation = separation_mutation(
            x_bounds, 
            distribution_info.output_min_x, distribution_info.output_max_x,
            distribution_info.separation, distribution_info.num_points,
            separation_alpha, separation_beta
        )
        if not new_separation:
            return
        
        distribution_info: DistributionInfo = offspring_solution.variables[variable_index]
        distribution_info.separation = new_separation
        distribution_info.output_min_x = new_out_min
        distribution_info.output_max_x = new_out_max
        offspring_solution.evaluated = False

class FixedMapCrossover(LocalVariator):
    """A LocalVariator for MonotonicDistributions 
    
    Choose offspring maps (candidate distributions) by crossing over parent maps."""
    
    _supported_types = MonotonicDistributions
    _supported_arity =  (2, None)
    _supported_noffspring =  (1, None)
    __slots__ = ("map_crossover_rate", "y_based_crossover", "equal_map_crossover")

    def __init__(self, 
                 map_crossover_rate = 0.1, 
                 y_based_crossover = False, 
                 equal_map_crossover = False):
        """
        Args:
            map_crossover_rate (float, optional): Defaults to 0.1.
            y_based_crossover (bool, optional): Indicates which domain to prioritize when converting output values to the new map. Defaults to False.
                - If True, keep the output start and end 'y' values as close as possible to the previous 'y' values during the conversion.
                - If False, keep the "relative" start and end 'x' values as close as possible to the previous relative 'x' values during the conversion
                -       "relative" = normalized to the width of the x domain (x_value / (x_max - x_min))
            equal_map_crossover (bool, optional): If True, the frequency of a map in parent solution is ignored. Defaults to False.
        """        
        self.map_crossover_rate = map_crossover_rate
        self.equal_map_crossover  = equal_map_crossover 
        self.y_based_crossover = y_based_crossover 
    
    def evolve(self, custom_type: MonotonicDistributions, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        crossover_rate = self.map_crossover_rate
        if not crossover_rate > 0 or custom_type.num_functions == 1:
            return
        equal_choice = self.equal_map_crossover
        y_based = self.y_based_crossover 
        
        parent_distrib_info = self.get_solution_variables(parent_solutions, variable_index)
        parent_maps = [distrib_info.map_index for distrib_info in parent_distrib_info]
        if equal_choice:
            parent_maps = list(set(parent_maps))
            if len(parent_maps) == 1:
                return
            
        if not custom_type.ordinal_maps:
            for i, offspring in enumerate(offspring_solutions):
                if np.random.uniform() < crossover_rate:
                    rand_map = np.random.choice(parent_maps)
                    distribution_info: DistributionInfo = offspring.variables[variable_index]
                    if rand_map != distribution_info.map_index:
                        FixedMapConversion._execute_map_conversion(custom_type, distribution_info, previous_map=distribution_info.map_index, new_map=rand_map, y_based=y_based)
                        offspring.evaluated = False       
        else:
            nmaps = len(parent_maps)
            nbits = _nbits_encode(0, custom_type.num_functions - 1)
            parent_encoded_maps = np.empty((nmaps, nbits), dtype = np.bool_)
            for j in range(nmaps):
                parent_encoded_maps[j] = int_to_gray_encoding(parent_maps[j], 0, custom_type.num_functions - 1, nbits)
            for i, offspring in enumerate(offspring_solutions):
                if np.random.uniform() < crossover_rate:
                    distribution_info: DistributionInfo = offspring.variables[variable_index]
                    prev_map = distribution_info.map_index
                    offspring_encoded_map = int_to_gray_encoding(prev_map, 0, custom_type.num_functions - 1, nbits)
                    offspring_encoded_map = single_binary_swap(parent_encoded_maps, offspring_encoded_map)
                    new_map = gray_encoding_to_int(0, custom_type.num_functions - 1, offspring_encoded_map)
                    if new_map == prev_map:
                        continue
                    
                    FixedMapConversion._execute_map_conversion(custom_type, distribution_info, prev_map, new_map, y_based)
                    offspring.evaluated = False
    
class DistributionBoundsPCX(LocalVariator):
    _supported_types = MonotonicDistributions
    _supported_arity =  (2, None)
    _supported_noffspring =  (1, None)
    __slots__ = ("distribution_pcx_rate", "bound_eta", "bound_zeta", "y_based_pcx")
    
    def __init__(self, distribution_pcx_rate = 0.25, bound_eta = 0.1, bound_zeta = 0.1, y_based_pcx = False):
        self.distribution_pcx_rate = distribution_pcx_rate
        self.bound_eta = np.float32(bound_eta)
        self.bound_zeta = np.float32(bound_zeta)
        self.y_based_pcx = y_based_pcx 
        
    def evolve(self, custom_type: MonotonicDistributions, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        pcx_rate = self.distribution_pcx_rate
        if not (pcx_rate > 0 and np.random.uniform() < pcx_rate):
            return
        eta = self.bound_eta
        zeta = self.bound_zeta
        y_based_pcx = self.y_based_pcx
        
        nparents = len(parent_solutions)
        parent_distributions: list[DistributionInfo] = self.get_solution_variables(parent_solutions, variable_index)

        copy_groups = self.group_by_copy(copy_indices)
        unlabeled = copy_groups.get(-1)
        new_nparents = nparents
        if unlabeled: # add "unlabeled" offspring as parents
            for offspring_idx in unlabeled:
                copy_groups[new_nparents] = [offspring_idx]
                new_nparents += 1
                parent_distributions.append(offspring_solutions[offspring_idx].variables[variable_index])
            del copy_groups[-1]
              
        # Gather info on width, points, and calculate y values if applicable
        all_fixed_width = True
        all_fixed_points = True
        min_max_y = []
        for i, distrib_info in enumerate(parent_distributions):
            bijection: RealBijection = custom_type.map_suite[distrib_info.map_index]
            x_bounds = bijection.point_bounds
            true_min_width = x_bounds.true_min_width
            true_max_width = x_bounds.true_max_width
            
            if true_min_width < true_max_width:
                all_fixed_width = False
            if not x_bounds.min_points == x_bounds.max_points:
                all_fixed_points = False
            if y_based_pcx:
                y1 = bijection.fixed_forward_map(distrib_info.output_min_x)
                y2 = bijection.fixed_forward_map(distrib_info.output_max_x)
                min_max_y.append((min(y1,y2), max(y1,y2)))
        
        # Put relative parent variables in np.ndarray 
        ncrossover_vars = 2 if all_fixed_width  else 3
        if all_fixed_points:
            ncrossover_vars -= 1
            
        parent_vars = np.empty((len(parent_distributions), ncrossover_vars), np.float32, order = 'C')
        y_width = custom_type.global_max_y - custom_type.global_min_y
        for i, distrib_info in enumerate(parent_distributions):
            bijection: RealBijection = custom_type.map_suite[distrib_info.map_index]
            x_bounds = bijection.point_bounds
            # Relative min val
            if not y_based_pcx: 
                parent_vars[i,0] = (distrib_info.output_min_x - x_bounds.lower_bound) / x_bounds.bound_width
            else:
                parent_vars[i,0] = (min_max_y[i][0] - custom_type.global_min_y) / y_width
                
            if ncrossover_vars == 1:
                continue
            
            # Relative points
            curr_idx = 1 
            if not all_fixed_points: 
                parent_vars[i,1] = _min_max_norm_convert(custom_type.global_min_points, custom_type.global_max_points, distrib_info.num_points, True)
                if ncrossover_vars == 2:
                    continue
                curr_idx += 1
            
            # Relative max val
            if not y_based_pcx: 
                parent_vars[i,curr_idx] = distrib_info.output_max_x / x_bounds.bound_width
            else:
                parent_vars[i,curr_idx] = (min_max_y[i][1] - custom_type.global_min_y) / y_width

        # Do PCX by "reference" parent
        parent_to_row = np.arange(new_nparents, dtype=np.uint16)
        parent_at_last_row = new_nparents - 1
        for parent_idx, offspring_indices in copy_groups.items():
            
            new_reference_row = parent_to_row[parent_idx]
            parent_vars[[new_reference_row, -1]] = parent_vars[[-1, new_reference_row]]
            parent_to_row[parent_idx] = new_nparents -1
            parent_to_row[parent_at_last_row] = new_reference_row
            parent_at_last_row = parent_idx
            # assert len(np.unique(parent_to_row)) == new_nparents

            new_vars = normalized_2d_pcx(parent_vars, len(offspring_indices), eta, zeta, randomize=False)
            for i, offspring_idx in enumerate(offspring_indices):
                
                offspring_sol = offspring_solutions[offspring_idx]
                offspring_sol.evaluated = False
                offspring_distrib = offspring_sol.variables[variable_index]
                offspring_map = offspring_distrib.map_index
                bijection = custom_type.map_suite[offspring_map]
                
                var_tuple: NormalizedOutput = self._create_var_tuple(new_vars[i], ncrossover_vars, all_fixed_points)
                # assert row_to_parent[-1] == parent_idx, f"actual parent index {copy_indices[offspring_idx] }"
                # if parent_idx < nparents:
                #     assert copy_indices[offspring_idx] == parent_idx
                # else:
                #     assert copy_indices[offspring_idx] == None, f"actual copy_index: {offspring_idx}"

                if y_based_pcx:
                    apply_y_bound_pcx(
                        var_tuple, 
                        offspring_distrib, 
                        bijection,
                        custom_type.global_min_points, 
                        custom_type.global_max_points,
                        custom_type.global_min_y,
                        custom_type.global_max_y)
                else:
                    apply_x_bound_pcx(
                        var_tuple, 
                        offspring_distrib,
                        bijection.point_bounds,
                        custom_type.global_min_points, 
                        custom_type.global_max_points)
                    
    @staticmethod 
    def _create_var_tuple(offspring_vars: np.ndarray, ncrossover_vars: int, all_fixed_points: bool):
        if ncrossover_vars == 1:
            return NormalizedOutput(start = offspring_vars[0])
        elif ncrossover_vars == 3:
            return NormalizedOutput(start = offspring_vars[0], points = offspring_vars[1], end = offspring_vars[2])
        elif not all_fixed_points:
            return NormalizedOutput(start = offspring_vars[0], points = offspring_vars[1])
        
        return NormalizedOutput(start = offspring_vars[0], end = offspring_vars[1])