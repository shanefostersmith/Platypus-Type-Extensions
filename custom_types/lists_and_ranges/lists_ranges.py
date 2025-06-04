import numpy as np
import inspect
from ._tools import _stepped_range_mutation, _type_equals, find_closest_val
from bisect import bisect_left, bisect_right
from ..core import CustomType, LocalMutator, LocalVariator, LocalGAOperator
from ..utils import (_min_max_norm_convert, _nbits_encode, int_to_gray_encoding, gray_encoding_to_int,
                     vectorized_to_norm, vectorized_from_norm)
from ..integer_methods.integer_methods import multi_int_crossover, int_cross_over, int_mutation, _cut_points
from ..real_methods.numba_differential import real_mutation, differential_evolve
from ..real_methods.numba_pcx import normalized_2d_pcx, normalized_1d_pcx

from platypus import Solution
from collections.abc import Sequence
from numba import njit


class Categories(CustomType):
    """`
    A CustomType representing categorical parameters that are **not** ordinal.
        (If mutating/evolving a categorical variable that has an order, an integer should be evolved instead - i.e. an index)
    
    Input categories of the same type must have a valid *equals* operation
    
    """    
    def __init__(
        self, 
        categories: Sequence, 
        local_variator = None,
        local_mutator = None):
        """
        The LocalVariator `CategoryCrossover` is created if "category_crossover_prob" is a number > 0 \n
        The LocalMutator `CategoryMutation` is created if "category_mutation_prob" is a number > 0
        
        If both `CategoryCrossover` and `CategoryMutation` are created, then a `LocalGAOperator` is created to combine the two.

        Args:
            categories (Sequence): A sequence of categories (must be indexable and length must be > 1)
        """        
        if not len(categories) > 1:
            raise TypeError("There must be at least 2 elements in the input sequence")
        self.categories = categories
        super().__init__(local_variator = local_variator, local_mutator=local_mutator)
            
    def rand(self):
        return np.random.choice(self.categories)

    def __str__(self):
        return f"Categories({', '.join(map(str, self.categories))})"

class CategoryMutation(LocalMutator):
    _supported_types = Categories
    __slots__ = ("category_mutation_prob")
    
    def __init__(self, category_mutation_prob = 0.1):
        """
        Args:
            category_mutation_prob (float, optional): Probablity of mutation. Defaults to 0.1.
                If number of categories == 2, then the "other" category is chosen for the offspring. Otherwise, a random category is chosen.
        """        
        self.category_mutation_prob = category_mutation_prob
    
    def mutate(self, custom_type: Categories, offspring_solution: Solution, variable_index, **kwargs):
        mutation_prob = self.category_mutation_prob
        if not np.random.uniform() < mutation_prob:
            return

        previous_category = offspring_solution.variables[variable_index]
        new_category = None
        mutated = False
        if len(custom_type.categories) == 2:
            new_category = custom_type.categories[1] if _type_equals(previous_category, custom_type.categories[0]) else custom_type.categories[0] 
            mutated = True
        else:
            new_category = np.random.choice(custom_type.categories)
            mutated = not _type_equals(new_category,previous_category)
        if mutated:
            offspring_solution.variables[variable_index] = new_category
            offspring_solution.evaluated = False
            
class CategoryCrossover(LocalVariator):
    """A LocalVariator for Categories
    
    Offspring categories are chosen from the parent Solutions' categories """
    _supported_types = Categories
    _supported_arity = (2, None)
    _supported_noffspring = (1, None)
    __slots__ = ("category_crossover_rate", "equal_crossover")
    
    def __init__(self, category_crossover_rate = 0.2, equal_crossover = False):
        """
        Args:
            category_crossover_rate (float, optional): Probability of choosing a random parent category for the offspring category. Defaults to 0.25.
                All offspring are evolved or none 
            equal_crossover (bool, optional): Defaults to False.
                If False, then probability of choosing a parent category during crossover is dependent on how many parents Solutions have that category.
                
                If True, all categories that exist in parent Solutions have an equal chance of being chosen (ignore the frequency of a category in parent solutions)
                    
        """        
        self.category_crossover_rate = category_crossover_rate
        self.equal_crossover = equal_crossover
    
    def evolve(self, custom_type: Categories, parent_solutions: list[Solution], offspring_solutions: list[Solution], variable_index, copy_indices, **kwargs):
        do_crossover = kwargs.get("crossover")
        crossover_probability = self.category_crossover_rate
        if not (do_crossover or np.random.uniform() < crossover_probability):
            return
        
        parent_categories = self.get_solution_variables(parent_solutions, variable_index)
        if self.equal_crossover:
            parent_categories = list(set(parent_categories))
            if len(parent_categories) == 1:
                only_category = parent_categories[0]
                for offspring in offspring_solutions:
                    if not _type_equals(only_category, offspring.variables[variable_index]):
                        offspring.variables[variable_index] = only_category
                        offspring.evaluated = False
                return
            
        for offspring in offspring_solutions:
            new_category = np.random.choice(parent_categories)
            if not _type_equals(new_category, offspring.variables[variable_index]):
                offspring.variables[variable_index] = new_category
                offspring.evaluated = False
        
        
class SteppedRange(CustomType): # TODO INIT
    """Evolve/mutate numbers that have a step value.
    
    For example:
        `lower_bound == 1`, `upper_bound == 4`, `step = 0.25`
        - Evolve numbers in the discrete range `(1, 1.25, 1.5, ..., 4)`
    
    The upper bound is inclusive if it is a valid "step"
    """     

    def __init__(
        self,  
        lower_bound, upper_bound, 
        step_value,
        local_variator = None,
        local_mutator = None):

        if step_value <= 0:
            raise ValueError("The step value must be greater than 0")

        if upper_bound < lower_bound + step_value:
            raise ValueError(
                "The upper bound must be at least a 'step' value greater than the lower bound"
            )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step = step_value
        self.max_step = int((self.upper_bound - self.lower_bound) // step_value)
        super().__init__(local_variator=local_variator,local_mutator=local_mutator)

    def rand(self):
        return self.lower_bound + (self.step * np.random.randint(self.max_step +1))

    def __str__(self):
        return f"SteppedRange: (lb = {self.lower_bound}, ub = {self.upper_bound} step = {self.step})"

class SteppedRangeCrossover(LocalVariator): # TODO OVERRIDES
    """ A LocalVariator for a SteppedRange
    
    Does an integer crossover of parent steps to create step values for the offspring Solution"""
    _supported_types = SteppedRange
    _supported_arity = (2, None)
    _supported_noffspring = (1, None)
    __slots__ = ("stepped_crossover_rate")
    
    def __init__(self, stepped_crossover_rate = 0.2):
        self.stepped_crossover_rate = stepped_crossover_rate
    
    def evolve(self, custom_type: SteppedRange, parent_solutions: list[Solution], offspring_solutions: list[Solution], variable_index, copy_indices, **kwargs):
        do_crossover = kwargs.get("crossover")
        crossover_prob = self.stepped_crossover_rate
        if not (do_crossover or np.random.uniform() < crossover_prob):
            return
        
        parent_values = self.get_solution_variables(parent_solutions, variable_index)
        noffspring = len(offspring_solutions)
        nbits = _nbits_encode(0, custom_type.max_step)
        parent_bits = np.empty((len(parent_solutions), nbits), np.bool_)
        for i, par in enumerate(parent_values):
            parent_step = int((par - custom_type.lower_bound) // custom_type.step)
            print(f"parent_step: {parent_step}")
            parent_bits[i] = int_to_gray_encoding(parent_step, 0, custom_type.max_step, nbits)
            
        if noffspring == 1:
            offspring_bits = int_cross_over(parent_bits)
            new_step = gray_encoding_to_int(0, custom_type.max_step, offspring_bits)
            new_value = custom_type.lower_bound + (new_step * custom_type.step)
            if not offspring_solutions[0].variables[variable_index] == new_value:
                offspring_solutions[0].variables[variable_index] = new_value
                offspring_solutions[0].evaluated = False
            return
        else:
            offspring_bits = multi_int_crossover(parent_bits, noffspring)
            for i, offspring in enumerate(offspring_solutions):
                new_step = gray_encoding_to_int(0, custom_type.max_step, offspring_bits[i])
                new_value = custom_type.lower_bound + (new_step * custom_type.step) 
                if not offspring.variables[variable_index] == new_value:
                    offspring.variables[variable_index] = new_value
                    offspring.evaluated = False
            
class MultiIntegerRange(CustomType):
    """Evolve/mutate multiple integers simultaneously, where each integer defines a lower and upper bound.
        Similar to optimizing multiple Integer types, but stored in a single array.
    
    Mutation occurs on an individual basis, while crossover can occur on both on an individual basis and over the set of all integers.
    
    The return type is a 1D numpy array of integers
    
    (see `MultiIntegerMutation`, `MultiIntegerCrossover` and `ArrayCrossover`)
    """
    # Future work: Add SteppedRange-like objects as valid inputs to this class
    
    def __init__(
        self, 
        ranges: np.ndarray, 
        dtype: type = np.int32,
        local_mutator: LocalMutator = None,
        local_variator: LocalVariator = None):             
        """
        If a LocalVariator and a LocalMutator are provided, a LocalGAOperator will be created. Equivalently, a LocalGAOperator can be provided as the LocalVariator.
        
        Args:
            ranges (np.ndarray): A 2D numpy array with `n` rows and 2 columns. The *inclusive* ranges of each integer
            
            - An integer `i` has a lower bound `inclusive_ranges[i,0]` and an inclusive upper bound `inclusive_ranges[i,1]`
            
            - `ranges[i,0]` may equal `ranges[i,1]`, in which case integer `i` is never mutated or evolved
                
            dtype (type): A numpy integer type: the type of the output arrays. Defaults to numpy.int32
                If a python `int` type is inputted, a numpy int32 will be used as default 
            
            local_mutator (LocalMutator, optional): A LocalMutator, should have MultiIntegerType registered in _supported_types. Defaults to None.
            local_variator (LocalVariator, optional): Cannot be a LocalMutator and shoud have MultiIntegerType registered in _supported_types. Defaults to None.
        """        
        if not (ranges.ndim == 2 and ranges.shape[0] > 0 and ranges.shape[1] == 2):
            raise TypeError("Invalid 'ranges' input: A 2D ndarray with at least 1 row and 2 columns is required")
        if dtype == int:
            dtype = np.int32
        elif not np.issubdtype(dtype, np.integer):
                raise TypeError(f"The 'dtype' must be a python int type or numpy integer type, got {dtype}")
        self.dtype = dtype
        self.ranges = ranges
        
        to_optimize = []
        for i in range(ranges.shape[0]):
            if ranges[i, 0] > ranges[i, 1]:
                raise ValueError(f"row {i}'s lower bound ({ranges[i, 0]} > the upper bound ({ranges[i, 1]}). The upper bound must be less than or equal to the lower bound)")
            if ranges[i, 0] < ranges[i, 1]:
                to_optimize.append(i)
        
        if not to_optimize:
            raise ValueError("All ranges in 'inclusive_ranges' are equal. No mutation or evolution can occur")
        if local_mutator is None and local_variator is None:
            raise ValueError("Neither a LocalMutator or LocalVariator were inputted")

        self.noptimized = len(to_optimize)
        self._to_optimize = range(ranges.shape[0]) if self.noptimized == ranges.shape[0] else tuple(to_optimize)
        super().__init__(local_variator=local_variator, local_mutator=local_mutator)

    def rand(self):
        nranges = self.ranges.shape[0]
        result = np.empty(nranges, self.dtype)
        if nranges == self.noptimized:
            for i in self._to_optimize:
                result[i] = np.random.randint(self.ranges[i,0], self.ranges[i,1] + 1) 
        else:
            opt_idx = 0
            for i in range(nranges):
                if opt_idx == self.noptimized:
                    result[i] = self.ranges[i,0]
                    continue
                curr_optimized_row = self._to_optimize[opt_idx]
                if i == curr_optimized_row:
                    result[i] = np.random.randint(self.ranges[i,0], self.ranges[i,1] + 1)  
                    opt_idx += 1
                else:
                    result[i] = self.ranges[i,0]
        return result
        
    def __str__(self):
        return f"MultiIntegerRange: ({len(self._to_optimize)} number of ranges"

     
class MultiIntegerMutation(LocalMutator):
    """A LocalMutator for a MultiIntegerRange
    
    Mutates integers individually with a bit flip"""
    _supported_types = MultiIntegerRange
    __slots__ = ("single_int_mutation_prob")

    def __init__(self, single_int_mutation_prob = 1.0):
        """
        Args:
            single_int_mutation_prob: Probability a single integer will be mutated. Defaults to 1.0
                If >= 1, mutation probability defaults to `1 / max(2, number of integers)`
        """        
        self.single_int_mutation_prob = single_int_mutation_prob
    
    def mutate(self, custom_type: MultiIntegerRange, offspring_solution, variable_index, **kwargs):
        mutation_prob = self.single_int_mutation_prob 
        if not mutation_prob:
            return
        if mutation_prob >= 1:
            mutation_prob = 1.0 / max(2.0, len(custom_type._to_optimize))
        
        one_mutated = False
        offspring_vals = offspring_solution.variables[variable_index]
        for row_idx in custom_type._to_optimize:
            if np.random.uniform() < mutation_prob:
                lb = int(custom_type.ranges[row_idx, 0])
                ub = int(custom_type.ranges[row_idx, 1])
                nbits = _nbits_encode(lb, ub)
                curr_bits = int_to_gray_encoding(int(offspring_vals[row_idx]), lb, ub, nbits)
                new_bits, mutated = int_mutation(curr_bits)
                if mutated:
                    one_mutated = True
                    offspring_vals[row_idx] = gray_encoding_to_int(lb, ub, new_bits, custom_type.dtype)
        if one_mutated:
            offspring_solution.evaluated = False
        

class MultiIntegerCrossover(LocalVariator):
    """A LocalVariator for MultiIntegerRange
    
    Crossover of individual integers in parent arrays"""
    _supported_types = MultiIntegerRange
    _supported_arity = (2, None)
    _supported_noffspring = (1, None)
    __slots__ = ("single_int_crossover_rate")
    
    def __init__(self, single_int_crossover_rate = 1.0):
        """
        Args:
             single_int_crossover_rate: Probability that a single integer will be evolved from crossing over parents integers.  Defaults to 1.0
                The probability gate is applied to integers individually (if threshold is met, that integer is evolved for all offspring)
        """        
        self.single_int_crossover_rate = single_int_crossover_rate
    
    def evolve(self, custom_type: MultiIntegerRange, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        indv_crossover_prob = self.single_int_crossover_rate
        if not indv_crossover_prob:
            return
        if indv_crossover_prob >= 1:
            indv_crossover_prob = 1.0 / max(2.0, len(custom_type._to_optimize))
            
        noffspring = len(offspring_solutions)
        nparents = len(parent_solutions)
        parent_arrays = self.get_solution_variables(parent_solutions, variable_index)
        for row_idx in custom_type._to_optimize:
            if not np.random.uniform() < indv_crossover_prob:
                continue
            
            lb = int(custom_type.ranges[row_idx, 0])
            ub = int(custom_type.ranges[row_idx, 1])
            nbits = _nbits_encode(lb, ub)
            parent_bits = np.empty((nparents, nbits), np.bool_)
            for i, par in enumerate(parent_arrays):
                parent_val = int(par[row_idx])
                parent_bits[i] = int_to_gray_encoding(parent_val, lb, ub, nbits)
            
            if noffspring == 1:
                offspring_array = offspring_solutions[0].variables[variable_index]
                offspring_bits = int_cross_over(parent_bits)
                new_val = gray_encoding_to_int(lb, ub, offspring_bits)
                if new_val != offspring_array[row_idx]:
                    offspring_array[row_idx] = new_val
                    offspring_solutions[0].evaluated = False
            else:
                offspring_bits = multi_int_crossover(parent_bits, noffspring)
                for i, offspring in enumerate(offspring_solutions):
                    offspring_array = offspring.variables[variable_index]
                    new_val = gray_encoding_to_int(lb, ub, offspring_bits[i])
                    if new_val != offspring_array[row_idx]:
                        offspring_array[row_idx] = new_val
                        offspring.evaluated = False

class MultiRealRange(CustomType):
    """Evolve/mutate multiple floats simultaneously, where each float defines a range of values.
        Similar to evolving multiple Real types, but stored in a single array.
    
    Mutation occurs on an individual basis, while crossover can occur on both on an individual basis and over the set of all floats.
    
    The return type is a 1D numpy array of floats

    """    
    def __init__(
        self, 
        ranges: np.ndarray,
        dtype: type = np.float32,
        local_variator: LocalVariator = None,
        local_mutator: LocalMutator = None
        ):
        
        assert ranges.ndim == 2, "'ranges' must be a 2D numpy array"
        assert ranges.shape[0] > 0 and not ranges.shape[1] < 2, "At least 1 row and 2 columns must exist in 'ranges'"
        assert inspect.isclass(dtype), f"'dtype' must be a type. Got {dtype!r}"
        if dtype == float:
            dtype = np.float32
        else:
            assert issubclass(dtype, np.floating), "The 'dtype' must be a python float type or numpy float type"
        self.dtype = dtype
        self.ranges = ranges.astype(np.float32, copy = False)
        for i in range(ranges.shape[0]):
            if ranges[i,0] >= ranges[i,1]:
                raise  ValueError(f"row {i}'s lower bound ({ranges[i, 0]} >= the upper bound ({ranges[i, 1]}). The upper bound must be greater than the lower bound)")
        
        super().__init__(local_variator=local_variator, local_mutator=local_mutator)
            
    def rand(self):
        num_ranges = self.ranges.shape[0]
        result = np.empty(num_ranges, self.dtype)
        for i in range(num_ranges):
            result[i] = np.random.uniform(self.ranges[i,0], self.ranges[i,1])
        return result
    
class MultiRealPM(LocalMutator):
    """A LocalMutator for a MultiRealRange
    
    Mutates floats individually with polynomial mutation (PM)"""
    _supported_types = MultiRealRange
    __slots__ = ("single_real_mutation_prob", "distribution_index")
    
    def __init__(self, single_real_mutation_prob = 0.1, distribution_index = 20.0):
        self.single_real_mutation_prob = single_real_mutation_prob
        self.distribution_index = distribution_index
    
    def mutate(self, custom_type: MultiRealRange, offspring_solution, variable_index, **kwargs):
        
        mutation_probability = self.single_real_mutation_prob
        if not mutation_probability:
            return
        num_floats = custom_type.ranges.shape[0]
        if mutation_probability >= 1.0:
            mutation_probability = 1.0 / max(2, num_floats)
        
        eps = np.finfo(custom_type.dtype).tiny 
        mutation_occured = False
        offspring_array = offspring_solution.variables[variable_index]
        for i in range(num_floats):
            if np.random.uniform() < mutation_probability:
                x = np.float32(offspring_array[i])
                lb = custom_type.ranges[i,0]
                ub = custom_type.ranges[i,1] - eps
                new_val = real_mutation(x, lb, ub, self.distribution_index)
                if new_val != offspring_array[i]:
                    mutation_occured = True
                    offspring_array[i] = new_val
                    
        if mutation_occured:
            offspring_solution.evaluated = False
  
class MultiDifferentialEvolution(LocalVariator):
    """A LocalVariator for a MultiRealRange
    
    Applies differential evolution to individual floats 
    """
    _supported_types = MultiRealRange
    _supported_arity = (4,4)
    _supported_noffspring = (1,1)
    __slots__ = ("real_crossover_rate", "real_step_size")
    
    def __init__(self, real_crossover_rate = 0.1, real_step_size = 0.25):
        self.real_crossover_rate = real_crossover_rate
        self.real_step_size = real_step_size
    
    def evolve(self, custom_type: MultiRealRange, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):

        crossover_rate = self.real_crossover_rate
        if not crossover_rate:
            return
        step_size = self.real_step_size
        
        nreals = custom_type.ranges.shape[0]
        parent_vals = self.get_no_copy_variables(parent_solutions, variable_index, copy_indices[0])
        offspring_range = offspring_solutions[0].variables[variable_index]
        nreals = custom_type.ranges.shape[0]
        irand = np.random.randint(nreals)
        
        for i in range(nreals):
            if i == irand or np.random.uniform() < crossover_rate:
                new_val = differential_evolve(
                    custom_type.ranges[i,0], custom_type.ranges[i,1], 
                    parent_vals[0][i], parent_vals[1][i], parent_vals[2][i], 
                    step_size)
                offspring_range[i] = new_val 
        
        offspring_solutions[0].evaluated = False

class MultiPCX(LocalVariator):
    _supported_types = MultiRealRange
    _supported_arity = (2,None)
    _supported_noffspring = (1,None)
    __slots__ = ("pcx_rate", "eta", "zeta")
    
    def __init__(self, pcx_rate = 0.25, eta = 0.2, zeta = 0.2):
        self.pcx_rate = pcx_rate
        self.eta = np.float32(eta)
        self.zeta = np.float32(zeta)
    
    def evolve(self, custom_type: MultiRealRange, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        pcx_rate = self.pcx_rate 
        if not np.random.uniform() < pcx_rate:
            return
        print(f"noffspring {len(offspring_solutions)}")
        eta = self.eta
        zeta = self.zeta
        nparents = len(parent_solutions)
        nreals = custom_type.ranges.shape[0]
        norm_parent_reals = np.empty((nparents, nreals), np.float32, order='C')
        for i, par in enumerate(parent_solutions):
            norm_parent_reals[i] = vectorized_to_norm(custom_type.ranges, par.variables[variable_index])
        
        
        parent_to_row = np.arange(nparents, dtype=np.uint16)
        row_to_parent = np.arange(nparents, dtype=np.uint16)
        unique_copies=  self.group_by_copy(copy_indices) # parent -> offspring copies of parent
        for parent_idx, offspring_indices in unique_copies.items():
            if parent_idx == -1:
                continue
            # Swap parent copy to "reference" row
            row_to_last = parent_to_row[parent_idx]
            norm_parent_reals[[row_to_last, -1]] = norm_parent_reals[[-1, row_to_last]]
            p_i, p_last = row_to_parent[i], row_to_parent[-1]
            row_to_parent[i], row_to_parent[-1] = p_last, p_i
            parent_to_row[p_i], parent_to_row[p_last] = nparents - 1, row_to_last
            
            offspring_values = vectorized_from_norm( # Evolve all offspring that were copied from parent_idx
                ranges = custom_type.ranges, 
                values = normalized_2d_pcx(norm_parent_reals, len(offspring_indices), eta, zeta, randomize=False),
                dtype = custom_type.dtype
            )
            for i, offspring_idx in enumerate(offspring_indices):
                curr_offspring = offspring_solutions[offspring_idx]
                curr_offspring.variables[variable_index] = offspring_values[i]
                curr_offspring.evaluated = False

        unlabeled_offspring = unique_copies.get(-1)
        if unlabeled_offspring: # Not copied from any solutions in parents_solutions
            temp = None
            for offspring_idx in unlabeled_offspring:
                curr_offspring = offspring_solutions[offspring_idx]
                offspring_norm = vectorized_to_norm(custom_type.ranges, curr_offspring.variables[variable_index])
                rand_row = np.random.randint(nparents)
                norm_parent_reals[[rand_row, -1]] = norm_parent_reals[[-1, rand_row]]
                temp = norm_parent_reals[-1]
                norm_parent_reals[-1] = offspring_norm
                
                curr_offspring.variables[variable_index] = vectorized_from_norm(
                    ranges = custom_type.ranges,
                    values = normalized_2d_pcx(norm_parent_reals, 1, eta, zeta, randomize=False)[0],
                    dtype = custom_type.dtype
                )
                curr_offspring.evaluated = False
                norm_parent_reals[-1] = temp

class RealList(CustomType):
    """
    Evolve/mutate a number from a sequence a numbers.
    
    Useful when numbers are spread out in a non-linear way.
    
    Values will sorted in ascending order at initialization, if not already
    """    
    def __init__(
        self, 
        real_list: list | tuple | np.ndarray,
        local_variator = None,
        local_mutator = None):
        
        assert len(real_list) > 1, "The input sequence must have more than 1 element"
        
        # Check if list is sorted. Sort if not
        sorted = True
        if isinstance(real_list, np.ndarray):
            assert real_list.ndim == 1, "The input ndarray must be 1 dimensional"
            sorted = np.all(real_list[:-1] <= real_list[1:])
        else:
            prev = -np.inf
            for i in real_list:
                if i <= prev:
                    sorted = False
                    break
                prev = i
            self.reals = None
        if not sorted:
            if isinstance(real_list, (list, np.ndarray)):
                real_list.sort()
            else:
                real_list = list(real_list).sort()
                real_list = tuple(real_list)
 
        self.reals = real_list
        super().__init__(local_variator=local_variator, local_mutator=local_mutator)

    def rand(self):
        return np.random.choice(self.reals)
    
class RealListPM(LocalMutator):
    """A LocalMutator for a RealList
    
    Uses polynomial mutation and finds the closest value"""
    _supported_types = RealList
    __slots__ = ("real_mutation_prob", "distribution_index")
    
    def __init__(self, real_mutation_prob= 0.1, distribution_index = 20):
        self.real_mutation_prob = real_mutation_prob
        self.distribution_index = np.float32(distribution_index) 
    
    def mutate(self, custom_type: RealList, offspring_solution, variable_index, **kwargs):
        mutation_prob = self.real_mutation_prob
        distribution_index = self.distribution_index
        
        if np.random.uniform() < mutation_prob:
            new_val = real_mutation(
                x = np.float32(offspring_solution.variables[variable_index]), 
                lb = np.float32(custom_type.reals[0]), 
                ub = np.float32(custom_type.reals[-1]), 
                distrib_idx=distribution_index)
            
            true_val = find_closest_val(custom_type.reals, new_val)
            if true_val != offspring_solution.variables[variable_index]:
                offspring_solution.variables[variable_index] = true_val
                offspring_solution.evaluated = False

class RealListDE(LocalVariator):
    """A LocalVariator for a RealList
    
    Applies differential evolution and finds closest value"""
    _supported_types = RealList
    _supported_arity = (4.4)
    _supported_noffspring = (1,1)
    __slots__ = ("crossover_rate", "step_size")
    
    def __init__(self, crossover_rate = 0.25, step_size = 0.25):
        self.crossover_rate = crossover_rate
        self.step_size = np.float32(step_size)
    
    def evolve(self, custom_type: RealList, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        do_crossover = kwargs.get("crossover")
        crossover_rate = self.crossover_rate
        non_copy_parents = self.get_no_copy_variables(parent_solutions, variable_index, copy_indices[0])
        if do_crossover or np.random.uniform() < crossover_rate:
            step_size = self.step_size
            new_val = differential_evolve(
                custom_type.reals[0], 
                custom_type.reals[-1],
                non_copy_parents[0],
                non_copy_parents[1],
                non_copy_parents[2],
                step_size)
            true_val = find_closest_val(custom_type.reals, new_val)
            if true_val != offspring_solutions[0].variables[variable_index]:
                offspring_solutions[0].variables[variable_index] = true_val
                offspring_solutions[0].evaluated = False

class RealListPCX(LocalVariator):
    """A LocalVariator for a RealList
    
    Applies PCX and finds closest value"""
    
    _supported_types = RealList
    _supported_arity = (2,None)
    _supported_noffspring = (1,None)
    __slots__ = ("pcx_rate", "eta", "zeta")

    def __init__(self, pcx_rate = 0.25, eta = 0.25, zeta = 0.25):
        self.pcx_rate = pcx_rate
        self.eta = np.float32(eta)
        self.zeta = np.float32(zeta)
    
    def evolve(self, custom_type: RealList, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs): #TODO find true val
        
        if not np.random.uniform() < self.pcx_rate:
            return
        eta = self.eta
        zeta = self.zeta
        nparents = len(parent_solutions)
        
        norm_parent_reals = np.empty(nparents, np.float32, order='C')
        for i, par in enumerate(parent_solutions):
            norm_parent_reals[i] = _min_max_norm_convert(custom_type.reals[0], custom_type.reals[-1], par.variables[variable_index], True)
        
        parent_to_row = np.arange(nparents, dtype=np.uint16)
        row_to_parent = np.arange(nparents, dtype=np.uint16)
        unique_copies=  self.group_by_copy(copy_indices) # parent -> offspring copies of parent
        for parent_idx, offspring_indices in unique_copies.items():
            if parent_idx == -1:
                continue
            
            # Swap parent copy to "reference" row
            row_to_last = parent_to_row[parent_idx]
            norm_parent_reals[[row_to_last, -1]] = norm_parent_reals[[-1, row_to_last]]
            p_i, p_last = row_to_parent[i], row_to_parent[-1]
            row_to_parent[i], row_to_parent[-1] = p_last, p_i
            parent_to_row[p_i], parent_to_row[p_last] = nparents - 1, row_to_last
            
            new_values = normalized_1d_pcx(norm_parent_reals, np.uint8(len(offspring_indices)), eta, zeta, randomize=False)
            for i, offspring_idx in enumerate(offspring_indices):
                curr_offspring = offspring_solutions[offspring_idx]
                true_val = find_closest_val(custom_type.reals, _min_max_norm_convert(custom_type.reals[0], custom_type.reals[-1], new_values[i], False))
                if true_val != curr_offspring.variables[variable_index]:
                    curr_offspring.variables[variable_index] = true_val
                    curr_offspring.evaluated = False
        
        unlabeled_offspring = unique_copies.get(-1)
        if unlabeled_offspring: # Not copied from any solutions in parents_solutions
            temp = None
            for offspring_idx in unlabeled_offspring:
                curr_offspring = offspring_solutions[offspring_idx]
                offspring_norm = _min_max_norm_convert(custom_type.reals[0], custom_type.reals[-1], curr_offspring.variables[variable_index], True)
                rand_row = np.random.randint(nparents)
                norm_parent_reals[[rand_row, -1]] = norm_parent_reals[[-1, rand_row]]
                temp = norm_parent_reals[-1]
                norm_parent_reals[-1] = offspring_norm
                
                new_value = normalized_1d_pcx(norm_parent_reals, np.uint8(1), eta, zeta, randomize=False)[0]
                true_val = find_closest_val(custom_type.reals, _min_max_norm_convert(custom_type.reals[0], custom_type.reals[-1], new_value, False))
                if true_val != curr_offspring.variables[variable_index]:
                    curr_offspring.variables[variable_index] = true_val
                    curr_offspring.evaluated = False
                    
                norm_parent_reals[-1] = temp
        
class StepMutation(LocalMutator): 
    """A LocalMutator for a SteppedRange or RealList.
    
    Mutates an offspring's value by incrementing up or down one step"""
    _supported_types = (SteppedRange, RealList)
    __slots__ = ("step_mutation_prob")
    
    def __init__(self, step_mutation_prob = 0.1):
        self.step_mutation_prob = step_mutation_prob
    
    def mutate(self, custom_type: SteppedRange | RealList, offspring_solution, variable_index, **kwargs):
        mutation_prob = self.step_mutation_prob
        if np.random.uniform() < mutation_prob:
            if isinstance(custom_type, SteppedRange):
                offspring_solution.variables[variable_index] = _stepped_range_mutation(custom_type.lower_bound, custom_type.step, custom_type.max_step, offspring_solution.variables[variable_index])
            elif isinstance(custom_type, RealList):
                left_idx = bisect_left(custom_type.reals, offspring_solution.variables[variable_index])
                right_idx = bisect_right(custom_type.reals, offspring_solution.variables[variable_index], lo = left_idx)
                if right_idx:
                    right_idx -= 1
                new_idx = None
                if left_idx < right_idx:
                    new_idx = max(len(custom_type.reals) - 1, right_idx + 1) if np.random.randint(2) or left_idx == 0 else left_idx - 1
                else:
                    new_idx = int(_stepped_range_mutation(0, 1, len(custom_type.reals) - 1, left_idx))
                offspring_solution.variables[variable_index] = custom_type.reals[new_idx]

            offspring_solution.evaluated = False

class ArrayCrossover(LocalVariator):
    """A LocalVariator for a MultiIntegerRange or a MultiRealRange
    
    Replace sections of offspring arrays with sections of parent arrays
    """
    _supported_types = (MultiIntegerRange, MultiRealRange)
    _supported_arity = (2, None)
    _supported_noffspring = (1, None)
    __slots__ = ("array_crossover_rate")
    
    def __init__(self, array_crossover_rate = 0.2):
        """
        Args:
            array_crossover_rate: Probability that an offspring array replaces sections  of parent arrays
                The probability gate is applied to individual offspring solutions
        """   
        self.array_crossover_rate = array_crossover_rate
    
    def evolve(self, custom_type: MultiIntegerRange | MultiRealRange, parent_solutions, offspring_solutions, 
               variable_index, copy_indices, **kwargs):
        nparents = len(parent_solutions)
        global_crossover_prob = self.array_crossover_rate
        if not global_crossover_prob:
            return
        
        full_array_size = custom_type.ranges.shape[0]
        for i, offspring in enumerate(offspring_solutions): 
            if not np.random.uniform() < global_crossover_prob:
                continue
            
            curr_offspring_array = offspring.variables[variable_index]
            segment_sizes = _cut_points(full_array_size, max(2, nparents + np.random.randint(-1,2)))
            j = 0
            non_copy_parents =  list(range(nparents))
            if copy_indices[i] is not None:
                non_copy_parents.pop(copy_indices[i])
            for segment in segment_sizes:
                rand_parent = parent_solutions[np.random.choice(non_copy_parents)]
                parent_array = rand_parent.variables[variable_index]
                curr_offspring_array[j:j+segment] = parent_array[j:j+segment]
                j += segment
            offspring.evaluated = False