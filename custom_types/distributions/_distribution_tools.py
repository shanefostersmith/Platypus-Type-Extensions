import numpy as np
from dataclasses import dataclass
from .real_bijection import RealBijection

@dataclass
class DistributionInfo:
    __slots__ = ("map_index", "num_points", "separation", 
                 "output_min_x", "output_max_x")
    map_index:  int
    num_points: int
    separation:   np.floating
    output_min_x: np.floating
    output_max_x: np.floating

    def __eq__(self, value):
        return isinstance(value, DistributionInfo) and (
            value.map_index == self.map_index and
            value.num_points == self.num_points and
            value.output_min_x == self.output_min_x and
            value.separation == self.separation)
  
    def __hash__(self):
        return hash((self.map_index, self.num_points, self.output_min_x, self.separation))
        
def ordered_y_bounds(bijection: RealBijection):
    """Returns tuple:
    
    y_min, y_max, max_first_y, min_last_y
    """
    first_y, second_y = bijection.left_y_bounds
    penult_y, last_y = bijection.right_y_bounds
    if not bijection.direction: # increasing func
        return first_y, last_y, second_y, penult_y
    else:
        return last_y, first_y, penult_y, second_y
    
def fast_point_addition(
    bijection: RealBijection,
    prev_distribution: np.ndarray, 
    diff_at_min_x: bool,
    point_difference: int,
    output_min_x, 
    output_max_x,
    num_points: int,
    separation,
    reverse: bool,
    make_copy: bool):
    """
    If only removing or adding points, can mutate quickly by avoiding most recalculations
    """        
    
    new_distribution = None
    at_first_idx = diff_at_min_x != reverse
    if point_difference < 0: #Removing points from one end
        new_distribution = prev_distribution if not make_copy else np.copy(prev_distribution)
        if at_first_idx : 
            new_distribution = new_distribution[abs(point_difference):]
        else: 
            new_distribution = new_distribution[:point_difference]
            
    else: 
        new_distribution = np.zeros(num_points + point_difference, bijection._return_type) 
        if diff_at_min_x:
            separation *= bijection.dtype(-1)
        forward_map = bijection.forward_function
    
        if at_first_idx: # Add points backwards from start
            new_distribution[point_difference:] = prev_distribution
            curr_x = output_min_x if diff_at_min_x else output_max_x
            for i in range(point_difference-1, -1, -1):
                curr_x += separation
                new_distribution[i] = bijection.dtype(forward_map(curr_x)) #TODO: Is casting necessary / less efficient?
                
        else: # Add points forward from end
            new_distribution[: num_points] = prev_distribution
            curr_x = output_min_x if diff_at_min_x else output_max_x
            for i in range(point_difference):
                curr_idx =  num_points + i
                curr_x += separation
                new_distribution[curr_idx] = bijection.dtype(forward_map(curr_x)) 

    return new_distribution