import numpy as np
from numba import njit
from bisect import bisect_left

def _type_equals(elem1, elem2):
        """Safe equals for different types"""
        return type(elem1) == type(elem2) and elem1 == elem2  
    
@njit
def _stepped_range_mutation(lower_bound, step_value, max_step, curr_value):
    curr_step = (curr_value - lower_bound) // step_value
    if curr_step == max_step:
        return curr_value - step_value
    elif curr_step == 0:
        return curr_value + step_value
    elif np.random.randint(2):
        return curr_value - step_value
    else:
        return curr_value + step_value
    

def find_closest_val(real_list, new_val):
    idx = bisect_left(real_list, new_val)
    if idx == len(real_list):
        return real_list[-1]
    elif idx == 0:
        return real_list[0]
    elif new_val - real_list[idx - 1] > real_list[idx] - new_val:
        return real_list[idx]
    else:
        return real_list[idx - 1]

def _group_by_copy(copy_indices) -> dict[int, list[int]]:
    unique_copies = {} 
    for i, copy_idx in enumerate(copy_indices):
        if copy_idx is None:
            unique_copies.setdefault(-1, []).append(i)
        else:
            unique_copies.setdefault(copy_idx, []).append(i)
    return unique_copies
