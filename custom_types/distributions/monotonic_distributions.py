# Future work / areas for exploration: 
    # 1. Batch GPU computing of output distributions (CUDA and Metal Shading)
    # 2. Unpacking mutations / crossovers into individual LocalMutation and LocalEvolutions
    # 3. Decorators/context managers for mappings that can handle vectorized operations
    # 4. More sophisticated caching of known x -> y mappings for full and partial arrays
    # 5. Optimizing the parameters of a map as well (probably a separate class)
    # 6. Multi-dimensional distributions (direction bounds instead of point bounds, covariance matrices, etc.)
    # 7. Decrease of probability terms over generations 
        # Different rates of decrease for different types of crossover/mutation, or different elements of distributions
    # 8. Gradual tightening of PointBounds over generations
    # 9. Other potential optimizations (
        # With only one map input, can bypass all of the bounds checks (no conversions). For many maps, check if all x or y bounds are the same at initialization
        # Greater use a Numba decorators (prange, stencils, parallel, etc.) and standardizing their flags
        # Using elements known numeric optimizations for very large arrays, parameter optimization, or semi-guidance (recontextualized for EP and GP objectives)
            #  Simulated annealing (SciPy dual-annealing, simanneal), BFGS, shared/continually updated 'velocity' term, greater focus of distribution shapes and spreads, etc.
            #  May require a new GlobalEvolution with shared/persistant memory (tracking "directions" that are promising, while keeping novelty of EP and GP searches)

import numpy as np
from collections.abc import Iterable
from collections import namedtuple
from .real_bijection import RealBijection
from .point_bounds import PointBounds, bound_tools
from ._distribution_tools import DistributionInfo
from ._mutation_tools import *
from ._crossover_tools import *
from ..real_methods.numba_pcx import normalized_2d_pcx
from ..real_methods.numba_differential import differential_evolve
from ..integer_methods.integer_methods import int_mutation, single_binary_swap
from ..utils import _min_max_norm_convert, clip, _nbits_encode, int_to_gray_encoding, gray_encoding_to_int
from ..core import CustomType, LocalMutator, LocalVariator

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
        - Used to inspect the function the converged upon post-optimization 
        
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
        **variator_overrides):
        """
        Args
        --------
        `
        
        **mappings** (Iterable[RealBijection]) An iterable of *RealBijection* objects:
        
        - The set of candidate distribution (that are monotonic)
               
        - The inverse mapping function and full 'x'-bounds are required to be set
        
        **ordinal_maps**: Defaults to False.

        - Indicates that the RealBijection objects are inputted in some sort of order
        
        - This order could be related to any property of the candidate distributions or their bounds (described by the *RealBijection* objects)
            
        **max_points** (int, optional): A default maximum number of samples in an output.  Defaults to 1000. 
        
        - If a *RealBijection*'s max_points boundis already set, then that value will be used. Otherwise, this default value will be used.
            
        **sample_count_mutation_probability** (float, optional): Defaults to 0.1. Probability of mutating the number of points in the distribution.
        
        - If a map's minimum number of points equals the map's maximum number of points, this is the probability of a "width" mutation 
            (by mutating the distance between adjacent points)
        
        **sample_count_mutation_limit** (int, optional): Defaults to None. A limit to the the number of samples that can be added to or removed from a distribution during a sample count mutation. 
        
        - If a map's minimum number of points equals the map's maximum number of points, then this defines a limit on a"width" mutation:
            *width change limit* = *(x_max - x_min) / (number of points)*  * *sample_count_mutation_limit*
        
        **shift_mutation_probability** (float, optional): Defaults to 0.1. Probability of shifting entire distribution along the x-axis. 
        
        **shift_alpha** (float, optional): Defaults to 1.5. A parameter for `np.random.beta(shift_alpha, shift_beta)`. Alters distribution of shift mutations. 
        
        **shift_beta** (int, optional): Defaults to 6. A parameter for `np.random.beta(shift_alpha, shift_beta`. Alters distribution of shift mutations. 
        
        **mapping_mutation_probability** (float, optional): Defaults to 0.1. Probability of mutating mapping functions 
            
        - i.e. applying relative x values of a previous solution to new distribution
        
        - May also be informed by previous y values if *y-based crossover* is set to True, or the order of the maps, if *ordinal_maps* is set to True

        **randomization_probability** (float, optional): Defaults to 0.0. Probability that a distribution will be completely randomized during a mutation (aka. probability of calling `rand()` during mutation). 
        
        **global_mutation_probability**: (float, optional): Defaults to 0.0. 
        
        - If 0.0, sample count, shift and mapping mutations occur independently and according to their individual probability values 
        
            - (see _sample_count_mutation_probability_ , _shift_mutation_probability_ , _mapping_mutation_probability_)
        
        - If > 0, then sample count, shift and mapping mutations will happen simultanously (all or none mutation). 
        
            - If a sample count, shift or mapping mutation's individual probability is 0.0, that mutation type will not occur regardless of the *global_mutation_probability*
            
        - Randomizations are always independent (see *randomization_probability*).
        
        **sort_ascending** (bool, optional): Defaults to True. Indictes whether to return distribution samples in ascending or descending order
        
        **variator_overrides**: Optional parameters overriding any variator 

        - Parameter keywords may include `step_size`, `eta`, `zeta`, `real_method`
        
        - These parameters may also be passed into `evolve()` directly from a `GlobalEvolution` object
        
            - Parameters passed directly to `evolve()` will override initialization values
        
        - See `evolve()` for more details
                
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
            if x_bounds.lower_bound >= x_bounds.upper_bound:
                raise ValueError(f"RealBijection {i} has the same value for the minimum and maximum 'x' bound.")
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
    
    def _add_max_points(self, bounds: bound_tools.BoundsState, default_max_points: int):
        """Adds default max points when using a BoundsState"""
        default_max_points = max(bounds.min_points, default_max_points)
        bound_tools._cascade_from_points(default_max_points, from_max_points=True)
        
    def rand(self):
        rand_function_idx = 0 if self.num_functions == 1 else np.random.randint(self.num_functions)
        bijection: RealBijection = self.map_suite[rand_function_idx] 
        x_bounds = bijection.point_bounds
        
        x_min, max_x_start = x_bounds.first_point_bounds
        min_x_end, x_max = x_bounds.last_point_bounds
 
        # Get start of random output
        output_start = None
        output_width = x_bounds.fixed_width
        true_min_width = x_bounds.true_min_width
        true_max_width = x_bounds.true_max_width
        output_start = x_min if x_min == max_x_start else np.random.uniform(x_min, max_x_start)

        # Get a random width of output
        if output_width is None:
            curr_max_width = min(x_max - output_start, true_max_width)
            curr_min_width = None
            if min_x_end is not None and output_start < min_x_end:
                curr_min_width = clip(min_x_end - output_start, true_min_width, curr_max_width)
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
    then a bit flip mutation is used. Otherwise, offspring maps are chosen randomly from parent maps.
    """

    _supported_types = (MonotonicDistributions)
    
    def __init__(self, map_conversion_rate = 0.1, y_based_map_conversion = False):
        self.map_conversion_rate = map_conversion_rate
        self.y_based_map_conversion = y_based_map_conversion
        
    def mutate(self, custom_type: MonotonicDistributions, offspring_solution, variable_index, **kwargs):
        mapping_mutation_prob = 0 if custom_type.num_functions == 1 else self.map_conversion_rate
        if not custom_type.do_mutation or not (mapping_mutation_prob > 0 and np.random.uniform() < mapping_mutation_prob):
            return

        y_based = self.y_based_map_conversion
        distribution_info: DistributionInfo = offspring_solution.variables[variable_index]

        new_map_idx = None
        if custom_type.num_functions == 2:
            new_map_idx = ~distribution_info.map_index
        elif custom_type.ordinal_maps: #TODO change to gray
            binary_map_idx = int_to_gray_encoding(distribution_info.map_index, 0, custom_type.num_functions - 1)
            new_map_bits, mutated = int_mutation(binary_map_idx)
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
    
    def __init__(self, x_shift_prob = 0.5, shift_alpha = 1.5, shift_beta = 6):
        if shift_alpha <= 0 or shift_beta <= 0:
            raise ValueError("Both 'shift_alpha' and 'shift_beta' must be greater than 0")
        self.x_shift_prob = x_shift_prob
        self.shift_alpha = shift_alpha
        self.shift_beta = shift_beta

    def mutate(self, custom_type: MonotonicDistributions, offspring_solution, variable_index, **kwargs):
        distribution_info: DistributionInfo = offspring_solution.variables[variable_index]
        bijection: RealBijection = custom_type.map_suite[distribution_info.map_index]
        x_bounds = bijection.point_bounds
        shift_mutation_prob = 0 if x_bounds.lower_bound == x_bounds.max_first_point or x_bounds.upper_bound == x_bounds.min_last_point else self.x_shift_prob
        if not custom_type.do_mutation or not (shift_mutation_prob > 0 and np.random.uniform() < shift_mutation_prob):
            return
        
        shift_alpha = self.shift_alpha
        shift_beta = self.shift_alpha
        x_shift = shift_mutation(
            x_bounds, 
            distribution_info.output_min_x, distribution_info.output_max_x, 
            shift_alpha, shift_beta, 
            return_type = bijection.dtype)
        
        if not x_shift:
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
    
    def __init__(self, sample_count_mutation_rate = 0.1, sample_count_mutation_limit: int | None = None):
        self.sample_count_mutation_rate = sample_count_mutation_rate
        if sample_count_mutation_limit is not None:
            sample_count_mutation_limit = int(max(1, sample_count_mutation_limit))
        self.sample_count_mutation_limit = None
    
    def mutate(self, custom_type: MonotonicDistributions, offspring_solution, variable_index, **kwargs):
        distribution_info: DistributionInfo = offspring_solution.variables[variable_index]
        bijection: RealBijection = custom_type.map_suite[distribution_info.map_index]
        x_bounds = bijection.point_bounds
        count_mutation_prob = self.sample_count_mutation_rate 
        
        if not custom_type.do_mutation or x_bounds.min_points == x_bounds.max_points or not (
            count_mutation_prob > 0 and np.random.uniform() < count_mutation_prob):
            return
        
        count_limit = self.sample_count_mutation_limit
        point_difference, diff_at_min_x, separation_change = count_mutation(
            x_bounds, distribution_info.output_min_x, distribution_info.output_max_x,
            distribution_info.separation, distribution_info.num_points, count_limit)
        
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
    
    def __init__(self, separation_mutation_rate, separation_alpha, separation_beta):
        self.separation_mutation_prob = separation_mutation_rate,
        self.separation_alpha = separation_alpha
        self.separation_beta = separation_beta
    
    def mutate(self, custom_type: MonotonicDistributions, offspring_solution, variable_index, **kwargs):
        distribution_info: DistributionInfo = offspring_solution.variables[variable_index]
        bijection: RealBijection = custom_type.map_suite[distribution_info.map_index]
        x_bounds = bijection.point_bounds
        separation_mutation_prob = self.separation_mutation_prob
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
    """A LocalVariator for MonotonicDistributions and subclasses
    
    Crossover parent maps to """
    
    _supported_types = MonotonicDistributions
    _supported_arity =  (2, None)
    _supported_noffspring =  (1, None)

    def __init__(self, 
                 map_crossover_rate = 0.1, 
                 y_based_map_crossover = False, 
                 equal_map_crossover = False):
        self.map_crossover_rate = map_crossover_rate
        self.equal_map_crossover  = equal_map_crossover 
        self.y_based_map_crossover = y_based_map_crossover
    
    def evolve(self, custom_type: MonotonicDistributions, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        crossover_rate = self.map_crossover_rate
        if not crossover_rate > 0 or custom_type.num_functions == 1:
            return
        equal_choice = self.equal_map_crossover
        y_based = self.y_based_map_crossover
        
        parent_distrib_info = self.get_solution_variables(parent_solutions, variable_index)
        parent_maps = [distrib_info.map_index for distrib_info in parent_distrib_info]
        if equal_choice:
            parent_maps = list(set(parent_maps))
            if len(parent_maps == 1):
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
            nparents = len(parent_solutions)
            nbits = _nbits_encode(0, custom_type.num_functions - 1)
            parent_encoded_maps = np.empty((nparents, nbits), dtype = np.bool_)
            for j in range(nparents):
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
        new_copies = copy_indices
        if unlabeled: # add "unlabeled" offspring as parents
            new_nparents = nparents
            new_copies.copy()
            for offspring_idx in unlabeled:
                new_copies[offspring_idx] = new_nparents 
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
            
            if not y_based_pcx: # Relative min val
                parent_vars[i,0] = (distrib_info.output_min_x - x_bounds.lower_bound) / x_bounds.bound_width
            else:
                parent_vars[i,0] = (min_max_y[i][0] - custom_type.global_min_y) / y_width
            if ncrossover_vars == 1:
                continue
            
            curr_idx = 1 
            if not all_fixed_points: # Relative points
                parent_vars[i,1] = _min_max_norm_convert(custom_type.global_min_points, custom_type.global_max_points, distrib_info.num_points, True)
                if ncrossover_vars == 2:
                    continue
                curr_idx += 1
            
            if not y_based_pcx: # Relative max val
                parent_vars[i,curr_idx] = distrib_info.output_max_x / x_bounds.bound_width
            else:
                parent_vars[i,curr_idx] = (min_max_y[i][1] - custom_type.global_min_y) / y_width

        
        # Do PCX by "reference" parent
        parent_to_row = np.arange(nparents, dtype=np.uint16)
        row_to_parent = np.arange(nparents, dtype=np.uint16)
        for parent_idx, offspring_indices in copy_groups.items():
            
            # Swap parent copy to "reference" row
            row_to_last = parent_to_row[parent_idx]
            parent_vars[[row_to_last, -1]] = parent_vars[[-1, row_to_last]]
            p_i, p_last = row_to_parent[i], row_to_parent[-1]
            row_to_parent[i], row_to_parent[-1] = p_last, p_i
            parent_to_row[p_i], parent_to_row[p_last] = nparents - 1, row_to_last
            
            new_vars = normalized_2d_pcx(parent_vars, len(offspring_indices), eta, zeta, randomize=False)
            for i, offspring_idx in enumerate(offspring_indices):
                offspring_sol = offspring_solutions[offspring_idx]
                offspring_sol.evaluated = False
                offspring_distrib = offspring_sol.variables[variable_index]
                var_tuple: NormalizedOutput = self._create_var_tuple(new_vars[i], ncrossover_vars, all_fixed_points)
                
                if y_based_pcx:
                    apply_y_bound_pcx(
                        var_tuple, 
                        offspring_distrib, 
                        custom_type.global_min_points, custom_type.global_max_points,
                        y_width)
                else:
                    apply_x_bound_pcx(
                        var_tuple, 
                        offspring_distrib,
                        custom_type.global_min_points, custom_type.global_max_points)
        
    def _create_var_tuple(offspring_vars: np.ndarray, ncrossover_vars: int, all_fixed_points: bool):
        if ncrossover_vars == 1:
            return NormalizedOutput(start = offspring_vars[0])
        if ncrossover_vars == 3:
            return NormalizedOutput(start = offspring_vars[0], points = offspring_vars[1], end = offspring_vars[2])
        if not all_fixed_points:
            return NormalizedOutput(start = offspring_vars[0], points = offspring_vars[1])
        return NormalizedOutput(start = offspring_vars[0], end = offspring_vars[1])