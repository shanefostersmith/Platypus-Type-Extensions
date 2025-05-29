import numpy as np
from numba import njit
from ..utils import _int_to_gray, _gray_to_int
from ..integer_methods.integer_methods import int_mutation

@njit
def fixed_subset_swap(
    offspring_directory: np.ndarray, 
    parent_inactive_indices: np.ndarray, 
    offspring_inactive_indices: np.ndarray, 
    num_inactive: int):
    
    evolved = False
    irand = np.random.randint(num_inactive)
    single_swap_probability = 1.0 / (num_inactive + 1.0)
    for i in range(num_inactive):
        offspring_inactive_idx = offspring_inactive_indices[i]
        parent_inactive_idx = parent_inactive_indices[i]
        if (offspring_directory[parent_inactive_idx] != -1 and # offspring not inactive at feature where parent is inactive
            offspring_directory[offspring_inactive_idx] == -1 and 
            i == irand or np.random.uniform() < single_swap_probability):
            
            temp = offspring_directory[parent_inactive_idx]
            offspring_directory[parent_inactive_idx] = -1
            offspring_directory[offspring_inactive_idx] = temp
            evolved = True
            
    return evolved

@njit
def one_offspring_active_crossover(parent_directories: np.ndarray, offspring_directory: np.ndarray, crossover_rate: float, nactives: int):
    evolved = False
    nparents = parent_directories.shape[0]
    nfeatures = parent_directories.shape[1]
    active_rand = np.random.randint(nactives)
    curr_active = 0
    look_at_next = False
    if nparents == 1:
        for feat in range(nfeatures):
            curr_bin = offspring_directory[feat]
            if curr_bin == -1:
                continue
            if not (curr_active == active_rand or look_at_next or np.random.uniform() < crossover_rate):
                curr_active += 1
                continue
            parent_bin = parent_directories[0,feat]
            if parent_bin == -1:
                look_at_next |= curr_active == active_rand
                curr_active += 1
                continue
            if parent_bin != curr_bin:
                offspring_directory[feat] = parent_bin
                evolved = True
            look_at_next = False
            curr_active += 1
        return evolved
    
    for feat in range(nfeatures):
        curr_bin = offspring_directory[feat]
        if curr_bin == -1:
            continue
        if not (curr_active == active_rand or look_at_next or np.random.uniform() < crossover_rate):
            curr_active += 1
            continue
        active_row_indices = np.where(parent_directories[:,feat] > -1)[0]
        nindices = len(active_row_indices)
        if nindices == 0: #parents not active
            look_at_next |= curr_active == active_rand 
            curr_active += 1
            continue  
        if nindices == 1 and parent_directories[active_row_indices[0],feat] != curr_bin:
            offspring_directory[feat] = parent_directories[active_row_indices[0],feat]
            evolved = True
        elif nindices > 1:
            rand_row_idx = np.random.choice(active_row_indices)
            new_bin = parent_directories[rand_row_idx,feat]
            if new_bin != curr_bin:
                offspring_directory[feat] = new_bin
                evolved = True
                
        look_at_next = False
        curr_active += 1
        
    return evolved

@njit
def multi_offspring_active_crossover(offspring_directories: tuple[np.ndarray], crossover_rate: float):
    """Assumes > 1 offspring directory"""
    noffspring = len(offspring_directories)
    nfeatures = len(offspring_directories[0])
    evolved = np.zeros(noffspring, np.bool_)

    look_at_next = False
    frand = np.random.randint(nfeatures)
    for feat in range(nfeatures):
        if not (frand == feat or look_at_next or np.random.uniform() < crossover_rate):
            continue
        swap_bin = -1
        original_bin = -1
        primary_offspring = 0
        found = 0
        for i, directory in enumerate(offspring_directories):
            curr_bin = directory[feat]
            if curr_bin == -1:
                continue
            if found == 0:
                found = 1
                swap_bin = curr_bin
                original_bin = curr_bin
                primary_offspring = i
                continue
            found += 1
            if swap_bin != curr_bin:
                offspring_directories[primary_offspring][feat] = curr_bin
                directory[feat] = swap_bin
                swap_bin = curr_bin
                evolved[i] = True
        if found <= 1:
            look_at_next |= frand == feat
            continue
        evolved[primary_offspring] = evolved[primary_offspring] | (offspring_directories[primary_offspring][feat] != original_bin)
    
    return evolved

@njit
def combined_set_mutation(subset_size: int, num_bins: int, num_inactives: int, directory: np.ndarray, mutate_permutation_probability: np.float32):
    """ Combines swapping inactive / active features with mutating active features (one less loop)

    Assumes:
            total_features > subset size > 1, mutate_permutation_probability > 0"""

    # print("here combined")
    total_features = len(directory)
    last_idx = np.random.randint(total_features)
    last_bin = num_bins - 1
    last_active = directory[last_idx] > -1
    seen_active = 0
    seen_inactive = 0 
    reverse_swap = False
    
    i_rand = 0 #random idx for a guaranteed inactive/active swap
    if last_active:
        if num_inactives > 1:
            i_rand = np.random.randint(num_inactives)
        if np.random.uniform() < mutate_permutation_probability:
            gray_encode, _ = _int_to_gray(directory[last_idx], 0, last_bin) 
            gray_encode, mutated = int_mutation(gray_encode, 0.0)
            if mutated:
                directory[last_idx] = _gray_to_int(0, last_bin, gray_encode)
            seen_active += 1   
    elif not last_active:
        reverse_swap = True
        i_rand = np.random.randint(subset_size)
    
    inactive_swap_probability = np.float32(1.0 / total_features)
    for i in range(total_features):
        # print(f"last: {directory[last_idx]} last_active? {last_active}, ")
        if i == last_idx:
            continue
        curr_elem = directory[i]
        curr_active = curr_elem > -1
        if not curr_active and last_active:
            if (not reverse_swap and i_rand == seen_inactive) or np.random.uniform() < inactive_swap_probability:
                directory[i] = directory[last_idx]
                directory[last_idx] = curr_elem 
                last_active = False
            seen_inactive += 1
                
        elif curr_active:
            if np.random.uniform() <  mutate_permutation_probability:
                gray_encode, _ = _int_to_gray(curr_elem, 0, last_bin)
                gray_encode, mutated = int_mutation(gray_encode, 0.0)
                if mutated:
                    curr_elem = _gray_to_int(0, last_bin, gray_encode)
                    directory[i] = curr_elem
            if not last_active: 
                if (reverse_swap and i_rand == seen_active) or np.random.uniform() < inactive_swap_probability:
                    directory[i] = directory[last_idx]
                    directory[last_idx] = curr_elem 
                    last_active = True
            seen_active += 1
        else:
            seen_inactive += 1

@njit
def active_set_mutation(subset_size: int, num_bins: int, directory: np.ndarray, mutate_permutation_probability: np.float32):
    """Assumes len(directory) > 1, num_bins > 1, mutate_permutation_probability > 0,
    
    Return bool indicating if a mutation occured"""
    # print("here active")
    total_features = directory.shape[0]
    last_bin = num_bins - 1
    
    seen_actives = 0
    i = 0
    did_mutation = False
    while seen_actives < subset_size and i < total_features:
        curr_elem = directory[i]
        if curr_elem == -1:
            i += 1
            continue
        if np.random.uniform() < mutate_permutation_probability:
            if num_bins == 2:
                directory[i] = 1 - curr_elem
                did_mutation = True
            else:
                gray_encode, _ = _int_to_gray(curr_elem, 0, last_bin)
                gray_encode, mutated = int_mutation(gray_encode, 0.0)
                if mutated:
                    did_mutation = True
                    directory[i] = _gray_to_int(0, last_bin, gray_encode)
                    
        seen_actives += 1
        i += 1
    
    # print(f"did mutation: {did_mutation}")
    return did_mutation

@njit
def subset_mutation(subset_size: int, num_inactives: int, directory: np.ndarray):
    """Assumes total_features > 2, subset size >= 2
    
    (worst case O(n), best O(1))
    
    TODO: Case where subset size << num_inactives"""
    # print("here subset")
    if num_inactives == 0:
        return
    
    total_features = len(directory)
    if num_inactives == 1 : # avoid trivial worst case
        if directory[-1] == -1: 
            rand_swap_idx = np.random.randint(total_features - 1)
            temp = directory[rand_swap_idx]
            directory[rand_swap_idx] = directory[-1] 
            directory[-1] = temp
            return
        elif directory[0] == -1:
            rand_swap_idx = np.random.randint(1,total_features)
            temp = directory[rand_swap_idx]
            directory[rand_swap_idx] = directory[0] 
            directory[0] = temp
            return

    swap_range = max(1.0, min(num_inactives, subset_size) // 2)
    first_idx = np.random.randint(total_features)
    random_width = np.random.randint(1, total_features - 1)
    irand = 1 if num_inactives == 1 else np.random.randint(1,min(num_inactives,subset_size) + 1)
    
    # Do swaps up to "swap_range" or iterate up to inactive number "irand"
    i = first_idx
    swaps = 0
    seen_inactives = 0
    earliest_j = -1
    previous_j = -1
    while True:
        j = (i - random_width) % total_features
        start_elem_active = directory[i] > -1
        end_elem_active = directory[j] > -1
        
        if not start_elem_active:
            seen_inactives += 1 
            if irand == seen_inactives:
                if end_elem_active:
                    swaps += 1
                    end_element = directory[j]
                    directory[j] = directory[i]
                    directory[i] = end_element 
                elif swaps == 0 and previous_j > -1: 
                    swaps += 1
                    final_j = previous_j if (previous_j == earliest_j or np.random.randint(2)) else earliest_j
                    end_element = directory[final_j]
                    directory[final_j] = directory[i]
                    directory[i] = end_element 
                else:
                    previous_j = j
                break
            
            elif end_elem_active:
                swaps += 1
                end_element = directory[j]
                directory[j] = directory[i]
                directory[i] = end_element 
            
        elif not end_elem_active:
            seen_inactives += 1
            swaps += 1
            end_element = directory[j]
            directory[j] = directory[i]
            directory[i] = end_element
            if irand == seen_inactives:
                break
            if earliest_j == -1:
                earliest_j = j
            previous_j = j
                
        else: # both active
            if earliest_j == -1:
                earliest_j = i
            previous_j = i

        i += 1
        if i == total_features:
            i = 0
        if swaps == swap_range or i == first_idx:
            break
        
    # Found two negative before swap occured or before postive could be found
    if swaps == 0:
        while True:
            i += 1
            if i == total_features:
                i = 0
            if directory[i] > -1:
                end_element = directory[previous_j]
                directory[previous_j] = directory[i]
                directory[i] = end_element
                break
            
@njit
def fix_weight(
    curr_weight: np.float32, 
    offspring_directory: np.ndarray, 
    min_weight: np.float32,
    max_weight: np.float32,
    active_indices: np.ndarray,
    avoid: int):
    """Fixes weights when one weight was added/removed to a WeightedSet"""
    if curr_weight == 1.0:
        return 
    if curr_weight > 1.0:
        weight_to_remove = curr_weight - 1.0
        for i in active_indices:
            curr_var = offspring_directory[i]
            dist_to_min = curr_var - min_weight
            if i == avoid or dist_to_min <= 0:
                continue
            if dist_to_min >= weight_to_remove:
                offspring_directory[i] = curr_var - weight_to_remove
                break
            else:
                offspring_directory[i] = min_weight
                weight_to_remove -= dist_to_min
    else:
        weight_to_add = 1.0 - curr_weight
        for i in active_indices:
            curr_var = offspring_directory[i]
            dist_to_max = max_weight - curr_var
            if dist_to_max <= 0:
                continue
            if dist_to_max >= weight_to_add:
                offspring_directory[i] = curr_var + weight_to_add
                break
            else:
                offspring_directory[i] = max_weight
                weight_to_add -= dist_to_max

@njit
def fix_all_weights(
    offspring_directory: np.ndarray, 
    curr_weight: np.float32,
    max_weight: np.float32,
    min_weight: np.float32,
    active_feature_indices: np.ndarray):
    """After WeightedSet crossover, adjust weights proportionally so sum of weight == 1

    Args:
        offspring_directory (np.ndarray): _description_
        curr_weight (np.float32): _description_
        max_weight (np.float32): _description_
        min_weight (np.float32): _description_
        active_feature_indices (np.ndarray): _description_
    """    
    if curr_weight == 1.0:
        return

    offspring_directory /= curr_weight
    if curr_weight < 1.0: # All elements were increased
        curr_max= np.max(offspring_directory)
        if curr_max <= max_weight:
            return
              
        # Decrease all weights over max -> (new_weight < 1)
        under_indices = np.empty(len(active_feature_indices), np.int32)
        dist_to_max = 0.0
        new_weight = 0.0
        i = 0
        for active_idx in active_feature_indices:
            weight = offspring_directory[active_idx]
            if weight >= max_weight:
                offspring_directory[active_idx] = max_weight
                new_weight += max_weight
                continue
            
            dist_to_max +=  max_weight - weight 
            new_weight += weight
            under_indices[i] = active_idx
            i += 1
        
        # Increase all active features (under max) by the same proportion
        under_multipiler = (1.0 - new_weight) / dist_to_max 
        for j in range(i):
            under_idx = under_indices[j]
            weight = offspring_directory[under_idx]
            curr_dist_to_max = max_weight - weight 
            weight += curr_dist_to_max * under_multipiler
            offspring_directory[under_idx] = weight
        
    else: # All elements were reduced
        curr_min = np.min(offspring_directory)
        if curr_min >= min_weight:
            return

        # Increase all weights under min -> (new_weight > 1)
        over_indices = np.empty(len(active_feature_indices), np.int32)
        dist_to_min = 0.0
        new_weight = 0.0
        i = 0
        for active_idx in active_feature_indices:
            weight = offspring_directory[active_idx]
            if weight <= min_weight:
                offspring_directory[active_idx] = min_weight
                new_weight += min_weight
                continue
            
            dist_to_min += weight - min_weight
            new_weight += weight
            over_indices[i] = active_idx
            i += 1
        
        # Decrease all active feature (above min) by the same proportion
        over_multipiler = (new_weight - 1.0) / dist_to_min
        for j in range(i):
            over_idx = over_indices[j]
            weight = offspring_directory[over_idx]
            curr_dist_to_min = weight - min_weight
            weight -= curr_dist_to_min * over_multipiler
            offspring_directory[over_idx] = weight
            
def basic_feature_crossover(parent_vars, total_feature_weights, 
                             min_weight, max_weight, true_max_weight, 
                             num_actives, total_features):
        """Used when the min subset size and max subset size are the same (or no size change), and no weight evolution occured

        Args:
            custom_type: A WeightedSet object
            parent_vars (np.ndarray): All the parent weights, stacked by row in matrix
            total_fetaure_weights (np.ndarray): The sum of parent weights, for every feature
        
        Returns:
            tuple (new directory, new_active_indices, new total_weight)
        """    
        new_active_indices = np.random.choice(np.arange(total_features), size = num_actives, replace = False, p = total_feature_weights / np.sum(total_feature_weights))
        counts = np.count_nonzero(parent_vars[:, new_active_indices], axis = 0)
        
        weight_range = max_weight - min_weight
        weight_adjustments = weight_range * np.random.beta(1, 10, size = num_actives)
        np.multiply(weight_adjustments, np.random.choice([-1.0, 1.0], size = num_actives), out = weight_adjustments)
        new_weights = [
            max(min_weight, min(true_max_weight, 
            total_feature_weights[new_active_indices[i]] / max(1.0, counts[i]) + weight_adjustments[i])) 
            for i in range(num_actives)
        ]
        curr_weight = sum(new_weights)
        all_weights = np.zeros(total_features, dtype = np.float32)
        all_weights[new_active_indices] = new_weights    
        
        return all_weights, new_active_indices, curr_weight

def add_new_features(
        offspring_directory, 
        new_features, 
        num_parents,
        active_weight_sum, 
        inactive_weight_sum,
        total_feature_weights,
        min_weight,
        max_weight,
        true_max_weight,
        ):
        
        ALPHA = 1.0
        BETA = 10.0
        num_new_features = len(new_features)
        if num_new_features == 1:
            rand_sign = -1.0 if np.random.randint(2) else 1.0
            new_weight = max(min_weight, 
                min(true_max_weight, 
                inactive_weight_sum / active_weight_sum + (rand_sign * np.random.beta(ALPHA,BETA)))
            )
            offspring_directory[new_features[0]] = new_weight 
            return  new_weight
        
        min_total_weight = num_new_features * min_weight
        max_total_weight = num_new_features * max_weight
        new_total_weight = max(min_total_weight, min(active_weight_sum, inactive_weight_sum / active_weight_sum, max_total_weight))
        
        new_weights = None
        weight_range = max_weight - min_weight
        weight_adjustments = weight_range * np.random.beta(ALPHA, BETA, size = num_new_features)
        if new_total_weight == min_total_weight:
            new_weights = min_weight + weight_adjustments
        elif new_total_weight == max_total_weight:
            new_weights = max_weight - weight_adjustments
        else:
            new_features = np.random.permutation(new_features)
            avg_weight = new_total_weight / np.float32(num_new_features)
            new_weights = np.full(num_new_features, avg_weight)
            for i in range(num_new_features - 1, 0, -1):
                weight = new_weights[i]
                other_idx = 0 if i == 1 else np.random.randint(i)
                other_weight = new_weights[other_idx]
                beta_val = weight_adjustments[i]
                if weight == min_weight or other_weight >= true_max_weight or weight < total_feature_weights[new_features[i]] / num_parents:
                    new_weights[i] = min(true_max_weight, weight + beta_val)
                    new_weights[other_idx] = max(min_weight, other_weight - beta_val)
                else:
                    new_weights[i] = max(min_weight, weight - beta_val)
                    new_weights[other_idx] = max(true_max_weight, other_weight + beta_val)
            
        offspring_directory[new_features] = new_weights
        return np.sum(new_weights)
    

# @njit
    # def _combined_swaps(subset_size: int, num_inactives: int, directory: np.ndarray, active_swap_probability: np.float32):
    #     """(experimental/untested: for swapping both inactive an active bins)"""
    #     total_features = len(directory)
    #     swap_range = max(1.0, min(num_inactives, subset_size) - 1)
    #     swap_probability = np.float32(1.0 / (swap_range + 1.0))
    #     rand_permutation = np.random.permutation(total_features)
        
    #     # Combined swaps between inactive/active and active/active (two pointer)
    #     end_pointer = total_features - 1
    #     start_pointer = 0
    #     attempted_swaps = 0
        
    #     while start_pointer < end_pointer: 
    #         start_idx = rand_permutation[start_pointer]
    #         end_idx = rand_permutation[end_pointer]
    #         start_active = directory[start_idx] > -1
    #         end_active = directory[end_idx] > - 1
            
    #         if start_active != end_active:
    #             if attemped_swaps < swap_range:
    #                 attemped_swaps += 1
    #                 if attempted_swaps == 1 or np.random.uniform() < swap_probability:
    #                     temp = directory[start_idx]
    #                     directory[start_idx] = directory[end_idx]
    #                     directory[end_idx]= temp
    #                     start_active = ~start_active
    #                     end_active= ~end_active
    #             start_pointer += ~start_active
    #             end_pointer -= ~end_active
    #             continue
            
    #         # both inactive
    #         if not start_active:
    #             start_pointer += 1
    #             if not attempted_swaps == 0: # want to do a least one inactive/active swap
    #                 end_pointer -= 1
    #             continue
            
    #         # both active
    #         if np.random.uniform() < active_swap_probability:
    #             temp = directory[start_idx]
    #             directory[start_idx] = directory[end_idx]
    #             directory[end_idx]= temp
    #         start_pointer += 1
    #         if not attemped_swaps < swap_range:
    #             end_pointer -= 1 
        