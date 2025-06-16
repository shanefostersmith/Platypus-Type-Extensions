import numpy as np
from typing import Optional
from  ._tools import *
from ..core import CustomType, LocalVariator, LocalMutator
from ..utils import _nbits_encode, int_to_gray_encoding, gray_encoding_to_int
from ..integer_methods.integer_methods import multi_int_crossover
from ..real_methods.numba_differential import real_mutation
from ..real_methods.numba_spx import _single_spx
from ..real_methods.numba_pcx import normalized_2d_pcx

class SetPartition(CustomType):
    """ `
    **Evolve/mutate a subset of elements partitioned into bins**

    Say you have `n` total features, a subset size of `x` and `y` number of bins (where `n` >= `x` and `n` >= `y`)
    
    - `x` elements are assigned to some number the `y` bins - a bin may have 0, 1 or multiple elements assigned to it at any time
    
    Return Type: A 1d numpy array of length `n` (and type int32)  - the **directory**
    
        - Each of the `n` features is assigned an index. Bins are assigned integers [0, y)
        
        - directory[`i`] == `j` indicates that feature `i` is assigned to bin `j`
        
        - directory[`i`] == `-1` indicates that feature `i` is *inactive* 
            (not placed in any of the `y` bins)
    `
    """   
    # Future work: if subset << total features, map active bins map to list features, dynamically update/remove keys 
    
    def __init__(
        self, 
        total_features: int, 
        subset_size: Optional[int] = None, 
        bins: Optional[int] = None, 
        equal_start = False,
        local_variator = None,
        local_mutator = None
    ):
        """ 
        Args:
            **total_features** (int): How many total features to choose from. Must be an integer >= 2
            
            **subset_size** (int | None, optional): How many of the total features may be placed in bins at any time. Defaults to None
                If None, then subset_size defaults to total_features
                Otherwise, subset_size must be an integer where 2 <= subset_size <= total_features
            
            **bins** (int | None, optional): How many bins the subset features may be placed into. Defaults to None.
                If bins is None, then bins defaults to subset_size.
                Otherwise, bins must be an integer where  2 <= bins <= total_features
                
            **equal_start**: Start with filling bins equally. Defaults to False.
                - If False, fully random `rand()` method.
                - If True, `rand()` function assigns features is bins as evenly as possible.
                    -       If bins == subset size, every bin has one feature. 
                    -       If bins > subset size, all features are in different bins, but not all bins have a feature.
                    -       If bins < subset size, bins with the least number of features will be filled first
                    
        """        
        
        self.total_features= int(total_features)
        if self.total_features < 2:
            raise ValueError(f"total_features must be greater than or equal to 2, got {self.total_features}")
        
        self.subset_size = None
        if subset_size is None:
            self.subset_size = self.total_features
        else:
            self.subset_size = int(subset_size)
            if not (2 <= self.subset_size <= total_features):
                raise ValueError(f"The subset_size must be in the range [2, total_features], got {self.subset_size}")
    
        self.num_bins = None
        if bins is None:
            self.num_bins = self.subset_size
        else:
            self.num_bins = int(bins)
            if not (2 <= self.subset_size <= total_features):
                raise ValueError(f"The number of bins must be in the range [2, total_features], got {self.num_bins}")

        self.equal_start = equal_start
        super().__init__(local_variator=local_variator, local_mutator=local_mutator)

    def rand(self):
        
        if self.total_features == self.subset_size: # no need to choose active features
            directory = np.empty(self.total_features, np.int32)
            if not self.equal_start or self.num_bins > self.total_features:
                directory[:] = np.random.choice(self.total_features, self.total_features, replace = (not self.equal_start))
            elif self.num_bins == self.total_features:
                directory[:] = np.random.permutation(self.total_features)
            else: # less bins than slots, and equal start
                i = 0
                while True:
                    j = min(self.total_features, i + self.num_bins)
                    curr_size = j - i
                    if curr_size == 1:
                        directory[i] = np.random.randint(self.num_bins)
                    else:
                        directory[i:j] = np.random.permutation(curr_size) if curr_size == self.num_bins else np.random.choice(self.num_bins, curr_size, replace=False)
                    if j >= self.total_features:
                        break
                    i = j
            return directory    
                        
        directory = np.full(self.total_features, -1, np.int32) 
        active_features =  np.random.choice(self.total_features, self.subset_size, replace = False)
        if not self.equal_start or self.num_bins > self.subset_size:
            directory[active_features] = np.random.choice(self.num_bins, self.subset_size, replace = (not self.equal_start))
        else:
            bins = np.arange(self.num_bins)
            if self.num_bins == self.subset_size:
                directory[active_features] = bins
                return directory
            i = 0
            while True: # less bins that active slots, and equal start
                j = min(self.subset_size, i + self.num_bins)
                curr_size = j - i
                if curr_size == 1:
                    directory[active_features[i]] = np.random.randint(self.num_bins)
                else:
                    directory[active_features[i:j]] = bins if curr_size == self.num_bins else np.random.choice(self.num_bins, curr_size, replace=False)
                if j >= self.subset_size:
                    break
                i = j 
                
        return directory
    
    def __str__(self):
        return f"FixedSubsetDistribution Subset Size: {self.subset_size}, Bins: {self.num_bins}"
    
    
class BinMutation(LocalMutator):
    """A LocalMutator for a SetPartition
    
    Swap active and inactive features of a directory and/or apply bit-flip mutation to the active feature bins.
    Depending on the probability gates, one, both, or neither of the mutations may occur"""
    _supported_types = SetPartition
    __slots__ = ("bin_mutation_probability", "bin_swap_rate")

    def __init__(self, bin_mutation_probability = 1.0,  bin_swap_rate = 0.1):
        """
        Args:
            bin_mutation_probability (float, optional): Probability of mutating an active feature's bin with a bit-flip. Defaults to 1.0.
                - Probability gate is applied to active features individually. \n
                - If probability >= 1, will default to `1 / num_active_features`
                
            bin_swap_rate (float, optional): The probability swapping an 'inactive' feature with an 'active' feature. Defaults to 0.1.
                - When the probability threshold is met, guarantees 1 swap will occur (with small probability other swaps will occur)\n
                - If a SetPartition's subset size is equal to the total number of features, then no swaps can occur.
        """     
        self.bin_mutation_probability = bin_mutation_probability
        self.bin_swap_rate = bin_swap_rate
    
    def mutate(self, custom_type: SetPartition, offspring_solution, variable_index, **kwargs):
        num_inactives = custom_type.total_features - custom_type.subset_size
        bit_flip_prob = self.bin_mutation_probability
        if bit_flip_prob >= 1:
            bit_flip_prob == 1.0 / custom_type.subset_size
        swap_rate = 0 if not num_inactives else self.bin_swap_rate
        
        mutated = False
        offspring_directory = offspring_solution.variables[variable_index]
        if swap_rate > 0 and np.random.uniform() < swap_rate:
            mutated = True
            if bit_flip_prob > 0:
                combined_set_mutation(custom_type.subset_size, custom_type.num_bins, num_inactives, offspring_directory, bit_flip_prob)
            else:
                subset_mutation(custom_type.subset_size, num_inactives, offspring_directory)
                
        if not mutated and bit_flip_prob > 0:
            mutated = active_set_mutation(custom_type.subset_size, custom_type.num_bins, offspring_directory, bit_flip_prob)
        
        if mutated:
            offspring_solution.evaluated = False
    
class FixedSubsetSwap(LocalVariator):
    """A LocalVariator for a SetPartition
    
    Swaps 'active' features with 'inactive' features. Decision is informed by which features are active or inactive in parent Solutions
    
    If the subset size of the SetPartition is the same as the total number of features, then no swap can occur.
    """
    
    _supported_types = SetPartition
    _supported_arity = (2,None)
    _supported_noffspring = (1,None)
    __slots__ = ("partition_swap_rate")
    
    def __init__(self, partition_swap_rate = 0.2):
        """
        Args:
            partition_swap_rate (float, optional): The probability an offspring solution will swap at least 1 active and inactive feature. Defaults to 0.2.
                The probability gate is applied to the offspring solutions individually
        """        
        self.partition_swap_rate = partition_swap_rate
           
    def evolve(self, custom_type: SetPartition, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        offspring_swap_probability = self.partition_swap_rate
        num_inactives = custom_type.total_features - custom_type.subset_size
        if not num_inactives or not offspring_swap_probability:
            return
        
        nparents = len(parent_solutions)
        for i, offspring in enumerate(offspring_solutions):
            if not np.random.uniform() < offspring_swap_probability:
                continue
            # choose a parent that is not a copy
            parent_swap_idx = np.random.choice(nparents)
            copied_from = copy_indices[i]
            if copied_from is not None and parent_swap_idx == copied_from:
                if parent_swap_idx == 0:
                    parent_swap_idx += 1
                elif parent_swap_idx == nparents - 1:
                    parent_swap_idx -= 1
                else:
                    parent_swap_idx += np.random.choice([-1,1])
                    
            parent_directory = parent_solutions[parent_swap_idx].variables[variable_index]
            parent_inactive_indices = np.random.permutation(np.where(parent_directory == -1)[0])
            offspring_directory = offspring.variables[variable_index]
            offspring_inactive_indices = np.where(offspring_directory == -1)[0]
            
            # Swap active and inactive features of offspring using parent inactive indices
            evolved = fixed_subset_swap(offspring_directory, parent_inactive_indices, offspring_inactive_indices, num_inactives)
            if evolved:
                offspring.evaluated = False
                
class ActiveBinSwap(LocalVariator):
    """A LocalVariator for a SetPartition
    
    Crossover the bins that 'active' features are assigned to (does nto change bin values). Inactive features are ignored
    """
    
    _supported_types = SetPartition
    _supported_arity = (2,None)
    _supported_noffspring = (1,None)
    __slots__ = ("active_swap_rate")
    
    def __init__(self, active_swap_rate = 0.2):
        """
        Args:
            active_swap_rate (float, optional): The probability the bin of an active feature will be evolved. Defaults to 0.2.
                The probability gate is applied to features individually. 
        """        
        self.active_swap_rate = active_swap_rate
   
    def evolve(self, custom_type: SetPartition, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        active_swap_rate = self.active_swap_rate
        if not active_swap_rate:
            return
        noffspring = len(offspring_solutions)
        if noffspring == 1:
            parent_directories = np.vstack(self.get_no_copy_variables(parent_solutions, variable_index, copy_indices[0]))
            evolved = one_offspring_active_crossover(parent_directories, offspring_solutions[0].variables[variable_index], active_swap_rate, custom_type.subset_size)
            if evolved:
                offspring_solutions[0].evaluated = False
        else:
            rand_order = np.random.permutation(noffspring)
            offspring_directories = tuple(offspring_solutions[i].variables[variable_index] for i in rand_order)
            evolved = multi_offspring_active_crossover(offspring_directories, active_swap_rate)
            for i, did_crossover in enumerate(evolved):
                if did_crossover:
                    offspring_idx = rand_order[i]
                    offspring_solutions[offspring_idx].evaluated = False


class WeightedSet(CustomType):
    """ `
    Evolve/mutute a subset of features, and a "weight" associated with each feature
    
    Say you have `n` total features, and a subset size in the range [`x,y`]
        (where `2` <= `x` <= `y` <= `n`)
    
        - A subset of the `n` total features will be chosen and a weight will be assigned to each feature

    Return Type: A 1d numpy array of length `n` (dype float32) - the **directory** 
    
        - Each of the `n` features is assigned an index. 
        
        - `sum(directory) ≈ 1` is always true
        
        - directory[`i`] == `j` > 0 indicates that feature `i` is a member of the subset with weight equal to `j`
        
        - directory[`i`] == 0 indicates that feature `i` is not included in the current subset     
        
    ` 
    """
    # Future work: decode when max_features << total features,
    
    def __init__(self, 
                 total_features: int, 
                 min_subset_size: int, 
                 max_subset_size: Optional[int] = None,
                 min_weight: float = 0.01,
                 max_weight: float = 0.99, 
                 local_variator = None,
                 local_mutator = None):
        """
        Args:
            **total_features** (int): An integer >= 2
            **min_subset_size** (int): An integer in inclusive range `[2, total_features]`
            **max_subset_size** (int | None, optional): An integer in inclusive range `[min_subset_size, total_features]`. Defaults to None
                - If None, the max_subset_size will default to total_features
            **min_weight** (float, optional): The minimum weight allowed for any one feature. Defaults to 0.001.
                - Must be a float in the exclusive range `(0.0, 1 / max_subset_size)`. Will be adjusted if >= 1 / *max_subset_size*)
            **max_weight** (float, optional): The maximum weight allowed for any one feature. Defaults to 0.99.
                - Must be greater than min_weight and in the exclusive range `(1 / min_subset_size, 1.0)`. Will be adjusted if <= 1 / min_subset_size
            **local_variator** (LocalVariator, optional): Cannot be a LocalMutator, should have WeightedSet registered in _supported_types. Defaults to None.
            **local_mutator** (LocalMutator, optional): A LocalMutator, should have WeightedSet registered in _supported_types. Defaults to None.
        
        Raises:
            ValueError: If total_features < 2
            ValueError: If not 2 <= min_subset_size <= total_features
            ValueError: If not min_subset_size <= max_subset_size <= total_features
            ValueError: if not 0 < min_weight < 1:
            ValueError: If not min_weight < max_weight < 1:
        """        
        
        # Set subset size bounds
        self.total_features = np.int32(total_features)
        if self.total_features < 2:
            raise ValueError(f"total_features must be greater than or equal to 2, got {self.total_features}")
        
        self.min_features = np.int32(min_subset_size)
        if not (2 <= self.min_features <= self.total_features):
            raise ValueError(
                f"The minimum subset size must be greater than 2 and less than the total "
                f"features ({self.total_features}), got {self.min_features}"
        )
        self.max_features = self.total_features
        if max_subset_size is not  None:
            self.max_features = np.int32(max_subset_size)
            if not (self.min_features <= self.max_features <= self.total_features):
                raise ValueError(
                    f"The max subset size must be >= the min subset size ({self.min_features}) "
                    f"and <= the total number of features ({self.total_features}), got {self.max_features}"
                )
        
        # Set weight bounds
        if not (0 < min_weight < 1):
            raise ValueError(f"The min weight must be greater than 0 and less than 1, got {min_weight}")
        self.min_weight = None
        self.max_weight = None
        if min_weight >= 1.0 / self.max_features:
            self.min_weight = np.float32(max(np.finfo(np.float32).tiny, 1.0 / (1.05 * self.max_features)))
        else:
            self.min_weight = np.float32(min_weight)
            
        if not (self.min_weight < max_weight < 1):
            raise ValueError(
                f"The max weight must be greater than the min_weight ({self.min_weight}) "
                f"and less than 1, got {max_weight}"
            )
        m_upper = 1.0 / self.min_features
        if max_weight <= m_upper:
            self.max_weight = np.float32(max(1.05 / self.min_features, min(1.0 - self.min_weight*self.min_features, max_weight + 1.05*(m_upper - max_weight))))
        else:
            self.max_weight = np.float32(max_weight)

        super().__init__(local_variator=local_variator, local_mutator=local_mutator)
        
    def rand(self):
        all_options = np.array(list(range(self.total_features)), dtype=np.int32)
        rand_subset_size = self.min_features if self.min_features == self.max_features else np.random.randint(self.min_features, self.max_features + 1)
        random_subset = np.random.choice(all_options, size = rand_subset_size, replace= False)
        random_weights = np.zeros(self.total_features, dtype=np.float32)
        
        true_max_weight = self._find_true_max_weight(rand_subset_size)  
        for i in random_subset:
            random_weights[i] = np.random.uniform(self.min_weight, self.max_weight)
        
        fix_all_weights(
            random_weights, 
            curr_weight=np.sum(random_weights),
            min_weight=self.min_weight,
            max_weight=true_max_weight,
            active_feature_indices = random_subset
        )
        return random_weights

    def _find_true_max_weight(self, curr_subset_size):
        return min(self.max_weight, np.float64(1.0 - self.min_weight*(curr_subset_size - 1.0)))
    
    def __str__(self):
        return f"Subset Weight Distribution: ({self.min_features} to {self.max_features} number of features"
    
class WeightedSetMutation(LocalMutator):
    """A LocalMutator for a WeightedSet
    
    Combines increasing or decreasing the subset size by one feature and mutating the weights associated with active features.
    """
    _supported_types = WeightedSet
    __slots__ = ("feature_mutation_probability", "weight_mutation_probability", "distribution_index")
    
    def __init__(
        self, 
        feature_mutation_probability = 0.1, 
        weight_mutation_probability = 0.1, 
        distribution_index = 20.0):
        """
        Args:
            feature_mutation_probability (float, optional): The probability that a feature will be randomly removed or added to the subset. Defaults to 0.1.
            weight_mutation_probability (float, optional): Probability that a pair of active features' weights are mutated. Defaults to 0.1.
                - One feature's weight is increased and the other is decreased (so that the sum of all weights still equals 1.0)
                - There is a small probability that other features' weights are mutated as well
            distribution_index (float, optional): Controls spread of weight mutations. Defaults to 20.0.
        """        
        self.feature_mutation_probability = feature_mutation_probability
        self.weight_mutation_probability= weight_mutation_probability
        self.distribution_index = distribution_index
    
    def mutate(self, custom_type: WeightedSet, offspring_solution, variable_index, **kwargs):
        
        feature_mutation_prob = 0 if custom_type.min_features == custom_type.max_features else self.feature_mutation_probability
        weight_mutation_prob = self.weight_mutation_probability
        if not (feature_mutation_prob or weight_mutation_prob):
            return
        distribution_index = self.distribution_index
        
        offspring_directory = offspring_solution.variables[variable_index]
        active_feature_indices = np.where(offspring_directory > 0)[0]
        num_active = len(active_feature_indices)
        mutated = False
        
        true_max_weight = None
        if feature_mutation_prob > 0 and np.random.uniform() < feature_mutation_prob:
            active_feature_indices = list(active_feature_indices)
            curr_weight = np.float32(1.0)
            added = -1
            # Increase subset size
            if num_active == custom_type.min_features or (num_active < custom_type.max_features and np.random.randint(2)): 
                inactive_features= np.setdiff1d(np.arange(custom_type.total_features), active_feature_indices, assume_unique=True)
                added = np.random.choice(inactive_features)
                active_feature_indices.append(added)
                num_active += 1
                active_features = offspring_directory[active_feature_indices]
                ub = min(custom_type._find_true_max_weight(num_active), np.float32(0.5))
                new_weight = _single_spx(active_features, custom_type.min_weight, ub)
                curr_weight += new_weight
                offspring_directory[added] = new_weight
                
            # Decrease subset size 
            else: 
                rand_active_idx = np.random.randint(num_active)
                rand_deleted_idx = active_feature_indices[rand_active_idx]
                num_active -= 1
                curr_weight -= offspring_directory[rand_deleted_idx]
                offspring_directory[rand_deleted_idx] = 0
                active_feature_indices.pop(rand_active_idx)

            # Fix weights
            mutated = True
            active_feature_indices = np.array(np.random.permutation(active_feature_indices), np.int32)
            true_max_weight = custom_type._find_true_max_weight(num_active)
            fix_weight(curr_weight, offspring_directory, custom_type.min_weight, true_max_weight, active_feature_indices, added)
        
        # Mutate weights
        if weight_mutation_prob > 0 and np.random.uniform() < weight_mutation_prob:
            if not mutated:
                active_feature_indices = np.random.permutation(active_feature_indices)
                true_max_weight = custom_type._find_true_max_weight(num_active)
            
            distribution_index = self.distribution_index
            real_probability = 1.0 / max(0.5, num_active - 1)
            for i in range(num_active - 1, 0, -1):
                if np.random.uniform() < real_probability or i == num_active - 1:
                    active_idx = active_feature_indices[i]
                    other_active_idx = active_feature_indices[0 if i == 1 else np.random.randint(i)]
                    prev_weight = offspring_directory[active_idx]
                    other_weight = offspring_directory[other_active_idx]
                    
                    p_ub = prev_weight + min(other_weight - custom_type.min_weight, true_max_weight - prev_weight)
                    p_lb = prev_weight - min(true_max_weight - other_weight, prev_weight - custom_type.min_weight)
                    new_weight = prev_weight if p_lb >= p_ub else real_mutation(prev_weight, p_lb, p_ub, distribution_index)
                    diff = new_weight - prev_weight
                    if diff != 0:
                        offspring_directory[active_idx] = new_weight
                        offspring_directory[other_active_idx] = other_weight - diff
                        mutated = True
                                  
        if mutated:
            offspring_solution.evaluated = False

class WeightedSetCrossover(LocalVariator):
    """A LocalVariator for a WeightedSet 
    
    Combines crossover of which features are active and crossover of the weights associated with active features"""
    _supported_types = WeightedSet
    _supported_arity = (2,None)
    _supported_noffspring = (1, None)
    __slots__ = ("subset_crossover_rate", "weight_crossover_rate", "eta", "zeta")
     
    def __init__(self, subset_crossover_rate = 0.2, weight_crossover_rate = 0.2, eta = 0.25, zeta = 0.25):
        """
        Args:
            subset_crossover_rate (float, optional): The probability of crossing over the active features of parent solutions. Defaults to 0.2.
                - The probability gate is applied to all offspring or none.
            weight_crossover_rate (float, optional): The probability of crossing over the weights of active features. Defaults to 0.2.
                - The probability gate is applied to the offspring individually
                - Weight values are crossed over with parent-centric crossover
            eta (float, optional): Controls distribution of weight crossovers. Defaults to 0.25.
            zeta (float, optional): Controls distribution of weight crossovers. Defaults to 0.25.
        """        
        self.subset_crossover_rate = subset_crossover_rate
        self.weight_crossover_rate = weight_crossover_rate 
        self.eta = np.float32(eta)
        self.zeta = np.float32(zeta)
    
    def evolve(self, custom_type: WeightedSet, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        evolve_subset_prob = self.subset_crossover_rate
        do_crossover = kwargs.get("crossover")
        do_feature_crossover = evolve_subset_prob > 0 and (do_crossover or np.random.uniform() < evolve_subset_prob) 
        weight_crossover_rate = self.weight_crossover_rate
        if not (do_feature_crossover or weight_crossover_rate):
            return
        
        nparents = len(parent_solutions)
        noffspring = len(offspring_solutions)
        eta = self.eta
        zeta = self.zeta
        
        parent_vars = np.empty((nparents, custom_type.total_features), dtype = np.float32, order = "C")
        for i, par in enumerate(parent_solutions):
            parent_vars[i] = par.variables[variable_index]
        total_feature_weights = np.sum(parent_vars, axis=0)
        
        # Find new subset sizes for offspring
        offspring_sizes = None
        if do_feature_crossover and not custom_type.min_features == custom_type.max_features:
            nbits = _nbits_encode(custom_type.min_features, custom_type.max_features)
            parent_subset_bits = np.empty((nparents, nbits), dtype = np.bool_)
            for i in range(nparents):
                parent_subset_bits[i] = int_to_gray_encoding(np.count_nonzero(parent_vars[i]), custom_type.min_features, custom_type.max_features, nbits)
            offspring_sizes = multi_int_crossover(parent_subset_bits, noffspring)
        
        active_features_indices = [None for _ in range(noffspring)]
        num_actives = [None for _ in range(noffspring)]
        curr_weights = [np.float32(1.0) for _ in range(noffspring)]
        parent_to_row = None if not weight_crossover_rate else np.arange(nparents, dtype = np.uint16)
        parent_at_last_row = nparents -1
        
        for i, offspring in enumerate(offspring_solutions):
            do_weight_crossover = weight_crossover_rate > 0 and (do_crossover or np.random.uniform() < weight_crossover_rate)
            if not (do_feature_crossover or do_weight_crossover):
                continue
            
            temp_directory = None
            curr_directory = offspring.variables[variable_index]
            offspring.evaluated = False

            if do_weight_crossover:
                # Swap parent copy to "reference" row
                parent_idx = copy_indices[i]
                if parent_idx is None:
                    parent_idx = np.random.randint(nparents)
                    temp_directory = parent_vars[parent_idx] # offspring directory as temporarily in the parent matrix
                    parent_vars[parent_idx] = curr_directory
                    
                if parent_idx < nparents - 1: 
                    new_reference_row = parent_to_row[parent_idx]
                    parent_vars[[new_reference_row, -1]] = parent_vars[[-1, new_reference_row]]
                    parent_to_row[parent_idx] = nparents -1
                    parent_to_row[parent_at_last_row] = new_reference_row
                    parent_at_last_row = parent_idx
                    # assert len(np.unique(parent_to_row)) == nparents, f"{parent_to_row}"
                    
            offspring.evaluated = False
            curr_directory = offspring.variables[variable_index]
            if do_weight_crossover and not do_feature_crossover: # No change in active features
                curr_active_indices = np.where(curr_directory > 0)[0]
                nactives = len(curr_active_indices)
                
                true_max_weight = custom_type._find_true_max_weight(nactives)
                offspring_weights = normalized_2d_pcx(parent_vars[:, curr_active_indices], 1, eta, zeta)[0]
                np.clip(offspring_weights, custom_type.min_weight, true_max_weight, out=offspring_weights)
                
                num_actives[i] = nactives
                active_features_indices[i] = curr_active_indices
                curr_weights[i] = np.sum(offspring_weights)
                curr_directory[curr_active_indices] = offspring_weights
                if temp_directory is not None:
                    parent_vars[-1] = temp_directory
                    temp_directory = None
                continue
            
            # Both crossovers
            if do_weight_crossover: 
                offspring_weights = normalized_2d_pcx(parent_vars, 1, eta, zeta)[0]
                curr_active_indices = None
                new_size = custom_type.min_features if custom_type.min_features == custom_type.max_features else gray_encoding_to_int(custom_type.min_features, custom_type.max_features, offspring_sizes[i])
                true_max_weight = custom_type._find_true_max_weight(new_size)
                
                # Remove 0 or 1 features 
                if new_size >= custom_type.total_features - 1:
                    min_idx = None
                    if new_size == custom_type.total_features - 1:
                        min_idx = np.argmin(offspring_weights)
                        
                    np.clip(offspring_weights, custom_type.min_weight, true_max_weight, out=offspring_weights)
                    curr_active_indices = list(range(custom_type.total_features))
                    if min_idx is not None:
                        offspring_weights[min_idx] = 0
                        curr_active_indices.pop(min_idx)
                    offspring.variables[variable_index] = offspring_weights
                
                # Remove > 1 feature, keep the highest weighted features
                else: 
                    curr_active_indices = np.argsort(offspring_weights)[-new_size:]
                    offspring_weights = offspring_weights[curr_active_indices]
                    np.clip(offspring_weights, custom_type.min_weight, true_max_weight, out=offspring_weights)      
                    all_weights = np.zeros(custom_type.total_features, np.float32)
                    all_weights[curr_active_indices] = offspring_weights 
                    offspring.variables[variable_index] = all_weights
                
                num_actives[i] = int(new_size)
                active_features_indices[i] = curr_active_indices
                curr_weights[i] = sum(offspring_weights)
                if temp_directory is not None:
                    parent_vars[-1] = temp_directory
                    temp_directory = None
                continue
            
            # subset crossover only
            new_features = []
            prev_size = np.count_nonzero(curr_directory)
            if offspring_sizes is not None:
                new_size = gray_encoding_to_int(custom_type.min_features, custom_type.max_features, offspring_sizes[i])
                true_max_weight = custom_type._find_true_max_weight(prev_size)
                
                if new_size == prev_size:
                    all_weights, new_active_indices, total_weight = basic_feature_crossover(
                        parent_vars, total_feature_weights, 
                        custom_type.min_weight, custom_type.max_weight, true_max_weight, 
                        prev_size, custom_type.total_features
                    )
                    offspring.variables[variable_index] = all_weights
                    active_features_indices[i] = new_active_indices
                    num_actives[i] = prev_size
                    curr_weights[i] = total_weight
                    continue
                
                prev_active_indices = list(np.where(curr_directory > 0)[0])
                active_weight_array = total_feature_weights[prev_active_indices] + np.finfo(np.float32).tiny
                active_weight_sum = np.sum(active_weight_array)
                
                # Remove features
                if new_size < prev_size: 
                    new_active_indices = np.random.choice(prev_active_indices, size = new_size, replace = False, p = active_weight_array / active_weight_sum)
                    removed_indices = np.setdiff1d(prev_active_indices, new_active_indices, assume_unique=True)
                    curr_weights[i] = -1
                    curr_directory[removed_indices] = 0
                    prev_active_indices = new_active_indices
                
                else: # Add Features
                    inactive_features= np.setdiff1d(np.arange(custom_type.total_features), prev_active_indices, assume_unique=True)
                    inactive_probabilities = total_feature_weights[inactive_features] + np.finfo(np.float32).tiny
                    inactive_weight_sum = np.sum(inactive_probabilities)
                    new_features = np.random.choice(inactive_features, size = new_size - prev_size, replace = False, p = inactive_probabilities / inactive_weight_sum)
                    
                    true_max_weight = custom_type._find_true_max_weight(new_size)
                    added_weight = add_new_features(
                        curr_directory, new_features, nparents, 
                        active_weight_sum, inactive_weight_sum, 
                        total_feature_weights,
                        custom_type.min_weight, custom_type.max_weight, true_max_weight
                    )
                    curr_weights[i] += added_weight
                    prev_active_indices.extend(new_features)
                    
                num_actives[i] = int(new_size)
                active_features_indices[i] = prev_active_indices
                
            else: # Feature crossover, no new sizes
                true_max_weight = custom_type._find_true_max_weight(prev_size)
                all_weights, new_active_indices, total_weight = basic_feature_crossover(
                    parent_vars, total_feature_weights, 
                    custom_type.min_weight, custom_type.max_weight, true_max_weight, 
                    prev_size, custom_type.total_features
                )
                offspring.variables[variable_index] = all_weights
                active_features_indices[i] = new_active_indices
                num_actives[i] = prev_size
                curr_weights[i] = total_weight
                
        
        # Fix weights
        for i, total_weight in enumerate(curr_weights):
            if num_actives[i] is None or total_weight == 1.0:
                continue
            elif total_weight == -1: # for removing lots of weight -> more precision
                total_weight = np.sum(offspring_solutions[i].variables[variable_index])
            
            true_max_weight = custom_type._find_true_max_weight(num_actives[i])
            # print(f"{i}: fix weight before: {total_weight}, {np.sum(offspring_solutions[i].variables[variable_index])}, true_max = {true_max_weight}, nactives: {num_actives[i]}")
            fix_all_weights(
                offspring_solutions[i].variables[variable_index], 
                total_weight, 
                true_max_weight,
                custom_type.min_weight,
                np.array(active_features_indices[i], dtype = np.int32))
            # print(f"fix weight after: {np.sum(offspring_solutions[i].variables[variable_index])}\n")
            