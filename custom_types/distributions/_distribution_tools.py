import numpy as np
from dataclasses import dataclass
from .real_bijection import RealBijection
from math import ceil

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
    
    def __repr__(self):
        return f"map_index: {self.map_index}, min: {self.output_min_x}, max: {self.output_max_x}, points: {self.num_points}, separation {self.separation}"
        
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


def sym_bound_adjustment(
    double_output_points,
    output_separation,
    curr_full_width,
    cardinality_bounds,
    separation_bounds,
    width_bounds,
    dtype,
    even_points: bool
):
    """Adjusts bounds for symmetric distributions. 
    Specify whether even or odd points are returned (whether or not center is included)

    Returns:
        tuple: (new_full_width, new_double_points, new_separation)

    """    
    full_min_width, full_max_width = width_bounds 
    full_min_points, full_max_points = cardinality_bounds
    full_min_separation, full_max_separation = separation_bounds
    
    in_point_bounds = full_min_points <= double_output_points <= full_max_points
    in_separation_bounds = full_min_separation <= output_separation <= full_max_separation
    in_width_bounds = full_min_width <= curr_full_width <= full_max_width
    
    assert even_points == bool(double_output_points % 2 == 0)
    if in_point_bounds and in_separation_bounds and in_width_bounds:
        return double_output_points, output_separation
    
    if not width_bounds:
        curr_full_width = full_min_width if curr_full_width < full_min_width else full_max_width
        output_separation = curr_full_width / dtype(double_output_points - 1)
        in_separation_bounds = full_min_separation <= output_separation <= full_max_separation
    
    if not in_point_bounds:
        if full_min_points == full_max_points:
            double_output_points = full_min_points
        elif double_output_points < full_min_points: # increase points, decrease separation
            double_output_points = full_min_points if even_points == bool(full_min_points % 2 == 0) else full_min_points + 1
            if output_separation > full_min_separation:
                output_separation = curr_full_width / dtype(double_output_points - 1)  
        else: # decrease points, increase separation
            double_output_points = full_max_points if even_points == bool(full_max_points % 2 == 0) else full_max_points - 1
            if output_separation < full_max_separation:
                output_separation = curr_full_width / dtype(double_output_points - 1)
            
        in_separation_bounds = full_min_separation <= output_separation <= full_max_separation
    
    if not in_separation_bounds:
        in_separation_bounds = True
        if output_separation > full_max_separation: # decrease separation, increase points
            output_separation = full_max_separation
            double_output_points = max(full_min_points, ceil(curr_full_width / full_max_separation + 1.0))
        else:
            output_separation = full_min_separation
            double_output_points = min(full_max_points, ceil(curr_full_width / full_min_separation + 1.0))
        if even_points != bool(double_output_points % 2 == 0):
            if double_output_points < full_max_points:
                double_output_points += 1
            else:
                double_output_points -= 1
            
    new_full_width = output_separation * dtype(double_output_points - 1)
    if not full_min_width <= new_full_width <= full_max_width:
        width_bound = full_min_width if  new_full_width < full_min_width else full_max_width
        temp_sep = width_bound / dtype(double_output_points - 1)
        if full_min_separation <= temp_sep <= full_max_separation:
            output_separation = temp_sep
        else:
            separation_bound = full_min_separation if temp_sep < full_min_separation else full_max_separation
            output_separation = separation_bound 
            double_output_points = max(full_min_points, min(full_max_points, ceil(width_bound / separation_bound + 1.0)))
            if even_points != bool(double_output_points % 2 == 0):
                if double_output_points == full_min_points or output_separation == full_max_separation:
                    double_output_points += 1
                else:
                    double_output_points -= 1
                output_separation = width_bound / dtype(double_output_points - 1)
    
    return double_output_points, output_separation
    
    
    
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