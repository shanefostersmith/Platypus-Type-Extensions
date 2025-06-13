from .real_bijection import RealBijection
from ._bounds_tools import BoundsViewMixin
from ..utils import _min_max_norm_convert, clip
from ._distribution_tools import *
from numba import njit

def map_conversion_x_based(prev_distribution: DistributionInfo, previous_bijection: RealBijection, new_bijection: RealBijection):
    """Returns:
        (new_min_x, new_max_x, new_separation, new_num_points)
    """
    prev_x_bounds = previous_bijection.point_bounds
    prev_x_min = prev_x_bounds.lower_bound
    prev_x_max = prev_x_bounds.upper_bound
    
    new_x_bounds = new_bijection.point_bounds
    true_min_width = new_x_bounds.true_min_width
    true_max_width = new_x_bounds.true_max_width
    new_width = None if true_min_width != true_max_width else true_min_width
    x_min, max_first_x = new_x_bounds.first_point_bounds
    min_last_x, x_max = new_x_bounds.last_point_bounds
    
    new_min_x = None
    new_max_x = None
    
    # convert starting x value
    prev_relative_x_min = _min_max_norm_convert(prev_x_min, prev_x_max, prev_distribution.output_min_x, True)
    relative_max_first = 0.0 if x_min == max_first_x else _min_max_norm_convert(x_min, x_max, max_first_x, True)
    prev_relative_x_min = min(relative_max_first, prev_relative_x_min)
    new_min_x = _min_max_norm_convert(x_min, x_max, prev_relative_x_min, False)
    
    # convert width
    if new_width is None:
        prev_width_relative = (prev_distribution.output_max_x - prev_distribution.output_min_x) / prev_x_bounds.bound_width
        new_full_width = new_x_bounds.bound_width
        temp_new_width = min(true_max_width, max(new_full_width * prev_width_relative, true_min_width))
        new_max_x = max(min_last_x, min(x_max, new_min_x + temp_new_width))
        new_width = new_max_x - new_min_x
    else: # fixed width
        new_max_x = new_min_x + new_width
    
    true_min_points, true_max_points = new_x_bounds.get_conditional_cardinality_with_width(new_width)
    new_num_points = min(true_max_points, max(true_min_points, prev_distribution.num_points))
    new_separation = new_width / new_x_bounds.dtype(new_num_points - 1)
    return new_min_x, new_max_x, new_separation, new_num_points

def map_conversion_y_based(
    prev_distribution: DistributionInfo, 
    prev_output_min_y: np.floating,
    prev_output_max_y: np.floating,
    new_bijection: RealBijection):
    """Returns:
        (new_min_x, new_max_x, new_separation, new_num_points)
    """
    
    num_points = prev_distribution.num_points
    
    new_x_bounds = new_bijection.point_bounds
    true_min_width = new_x_bounds.true_min_width
    true_max_width = new_x_bounds.true_max_width
    new_width = None if true_min_width != true_max_width else true_min_width

    x_min, max_first_x = new_x_bounds.first_point_bounds
    min_last_x, x_max = new_x_bounds.last_point_bounds
    y_min, y_max, max_first_y, min_last_y = ordered_y_bounds(new_bijection)
    decreasing = new_bijection.direction

    new_min_x = None
    new_max_x = None
    new_min_y = None 
    # Find new minimum y 
    if not decreasing: # (minimum y maps to minimum x)
        if prev_output_min_y <= y_min:
            new_min_y = y_min
            new_min_x = x_min
        elif prev_output_min_y < max_first_y:
            new_min_y = prev_output_min_y
            new_min_x = new_bijection.fixed_inverse_map(prev_output_min_y)
        else:
            new_min_y = max_first_y
            new_min_x = max_first_x
            if max_first_x == x_max - true_min_width:
                new_width = true_min_width
    else: # (minimum y maps to maximum x)
        if prev_output_min_y <= y_min:
            new_min_y = y_min
            new_max_x = x_max
        elif prev_output_min_y < max_first_y:
            new_min_y = prev_output_min_y
            new_max_x = new_bijection.fixed_inverse_map(prev_output_min_y)
        else:
            new_min_y = max_first_y
            new_max_x = min_last_x
            if min_last_x == x_min + true_min_width:
                new_width = true_min_width
    
    # Find new x width : Use previous "y-width"   
    if new_width is None:     
        previous_y_width = prev_output_max_y - prev_output_min_y
        temp_max_y = new_min_y + previous_y_width
        if temp_max_y > y_max:
            if new_min_x is None: 
                new_min_x = x_min
            else: 
                new_max_x = x_max
        elif temp_max_y > min_last_y:
            new_x = new_bijection.fixed_inverse_map(temp_max_y)
            if new_min_x is None:
                new_min_x = new_x
            else: 
                new_max_x = new_x
        else:
            if new_min_x is None:
                new_min_x = max_first_x
            else: 
                new_max_x = min_last_x
                    
        if not decreasing and new_max_x - new_min_x < true_min_width:
            new_max_x = new_min_x + true_min_width
        elif new_max_x - new_min_x < true_min_width:
            new_min_x = new_max_x - true_min_width

        new_width = new_max_x - new_min_x
        
    # (for fixed width or min width)
    elif new_min_x is None:
        new_min_x = new_max_x - new_width
    elif new_max_x is None:
        new_max_x = new_min_x + new_width

    assert new_min_x < new_max_x, f"Error in y-based map conversion, new max x {new_max_x,} not larger than new min x {new_min_x}. Could be caused by an incorrect setting of direction"  
            
    # get points / separations
    true_min_points, true_max_points = new_x_bounds.get_conditional_cardinality_with_width(new_width)
    new_num_points = min(true_max_points, max(true_min_points, num_points))
    new_separation = new_width / new_x_bounds.dtype(new_num_points - 1)
    return new_min_x, new_max_x, new_separation, new_num_points

np.random.seed(123)
def shift_mutation(x_bounds: BoundsViewMixin, output_min_x, output_max_x, shift_alpha, shift_beta, return_type):
        max_negative_shift = max(0.0, min(output_min_x - x_bounds.lower_bound, output_max_x - x_bounds.min_last_point))
        max_positive_shift = max(0.0, min(x_bounds.upper_bound - output_max_x , x_bounds.max_first_point - output_min_x))
        
        # Negative shift
        x_shift = 0.0
        if max_negative_shift > 0 and (not max_positive_shift > 0 or np.random.randint(2) == 1):
            beta = clip(np.random.beta(shift_alpha, shift_beta), return_type(0.0), return_type(1.0))
            x_shift = -beta * max_negative_shift
            
        # Positive shift
        elif max_positive_shift > 0:
            beta = clip(np.random.beta(shift_alpha, shift_beta), return_type(0.0), return_type(1.0))
            x_shift = beta * max_positive_shift
        
        return x_shift

def separation_mutation(
    x_bounds: BoundsViewMixin,
    output_min_x, output_max_x, 
    separation, curr_num_points,
    separation_alpha, separation_beta):
    """Returns:
        (new_out_min_x, new_out_max_x, new_separation)"""

    x_min, max_first_x = x_bounds.first_point_bounds
    min_last_x, x_max = x_bounds.last_point_bounds
    curr_width = (output_max_x - output_min_x)

    if x_bounds.min_separation == x_bounds.max_separation:
        return output_min_x, output_max_x, None
    
    true_min_width = x_bounds.true_min_width
    true_max_width = x_bounds.true_max_width
    flt_points = x_bounds.dtype(curr_num_points - 1)
    dist_from_max_sep = max(0.0, x_bounds.max_separation - separation) * flt_points
    dist_from_min_sep = max(0.0, separation - x_bounds.min_separation) * flt_points
    dist_from_max = max(0.0, min(true_max_width - curr_width, dist_from_max_sep))
    dist_from_min = max(0.0, min(curr_width - true_min_width, dist_from_min_sep))
    
    
    # Determine maximum width increase on each 'side' of current distribution
    min_side_dist = 0.0
    max_side_dist = 0.0
    width_change = 0.0
    if dist_from_max > 0 and (dist_from_min == 0 or np.random.randint(2)):
        min_side_dist = output_min_x - x_min
        max_side_dist = x_max - output_max_x
        dist_from_max = min(dist_from_max, min_side_dist + max_side_dist)
        width_change = dist_from_max * np.random.beta(separation_alpha, separation_beta)
    elif dist_from_min > 0:
        min_side_dist = max_first_x - output_min_x
        max_side_dist = output_max_x - min_last_x
        dist_from_min = min(dist_from_min, min_side_dist + max_side_dist)
        width_change = -1.0 * dist_from_min * np.random.beta(separation_alpha, separation_beta)
    
    if width_change == 0:
        return output_min_x, output_max_x, None
    
    # print(f"width_change: {width_change}, min_side_dist {min_side_dist}, max_side_dist {max_side_dist}")
    
    new_out_min =  output_min_x
    new_out_max =  output_max_x
    abs_width_change = abs(width_change)
    if min_side_dist <= 0:
        new_out_max += width_change
    elif max_side_dist <= 0:
        new_out_min -= width_change
    elif abs_width_change <= min_side_dist and abs_width_change <= max_side_dist:
        one_side_change = np.random.uniform() * width_change
        new_out_min -= one_side_change 
        new_out_max += (width_change - one_side_change)
    else: # Higher change of give more weight to the side w/ more room
        to_min = 0
        to_max = 0
        if min_side_dist > max_side_dist:
            max_side_proportion = max_side_dist / (max_side_dist + min_side_dist)
            rand_proportion = np.random.uniform(max_side_proportion, 1.0)
            to_min = min(abs_width_change * rand_proportion, min_side_dist) 
            to_max = min(abs_width_change - to_min, max_side_dist) 
        else:
            min_side_proportion = min_side_dist / (max_side_dist + min_side_dist)
            rand_proportion = np.random.uniform(min_side_proportion, 1.0)
            to_max = min(abs_width_change * rand_proportion, max_side_dist) 
            to_min = min(abs_width_change - to_max, min_side_dist) 
            
        if width_change < 0:
            new_out_min += to_min
            new_out_max -= to_max
        else:
            new_out_min -= to_min
            new_out_max += to_max
        
    new_separation = (new_out_max - new_out_min) / flt_points
    return new_out_min, new_out_max, new_separation

def count_mutation(all_x_bounds: BoundsViewMixin,
                    output_min_x, output_max_x, 
                    separation, curr_num_points,
                    count_limit):
    """
    Remove or add points from distribution. 
    
    If there is a fixed 'x' width (or close to it), spacing between x values changes instead

    Returns:
        tuple(int, bool): (point_diff, diff_at_min_x, separation_change)
            i.e. (number points added/removed, if adding or removing points from start x or last x, if changed separation instead of width)
    """        
    
    
    true_min_width = all_x_bounds.true_min_width
    true_max_width = all_x_bounds.true_max_width
    # Fixed width -> separation change with/ point change
    if true_max_width - true_min_width < separation: 
        width = output_max_x - output_min_x
        true_min_points, true_max_points = all_x_bounds.get_conditional_cardinality_with_width(width)
        max_addition = max(0, true_max_points - curr_num_points)
        max_subtraction = max(0,curr_num_points - true_min_points)
        if count_limit:
            max_addition = min(count_limit, max_addition)
            max_subtraction= min(count_limit, max_subtraction)
            
        point_diff = 0
        if max_subtraction and (not max_addition or np.random.randint(2)):
            point_diff = 1 if max_subtraction == 1 else np.random.randint(1, max_subtraction + 1)
            return -point_diff, False, width / all_x_bounds.dtype(curr_num_points - point_diff - 1)
        elif max_addition:
            point_diff = 1 if max_addition == 1 else np.random.randint(1, max_addition + 1)
            return point_diff, False, width / all_x_bounds.dtype(curr_num_points + point_diff - 1)
        
        return 0, False, None

    
    # Variable width -> num point change  
    x_min, max_first_x = all_x_bounds.first_point_bounds
    min_last_x, x_max = all_x_bounds.last_point_bounds
    curr_width = (output_max_x - output_min_x)
    true_min_points, true_max_points = all_x_bounds.get_conditional_cardinality_bounds(separation)
    if true_min_points == true_max_points:
        return 0, False, None

    # Check whether adding and subtracting points is valid given bounds
    addition_valid = (
        curr_num_points < true_max_points and
        curr_width + separation <= true_max_width and
        (output_max_x + separation <= x_max or output_min_x - separation >= x_min))
    subtraction_valid = (
        curr_num_points > true_min_points and 
        curr_width >= true_min_width + separation and
        (output_min_x + separation <= max_first_x or output_max_x - separation >= min_last_x))
    if not (addition_valid or subtraction_valid):
        return 0, False, None
    
    if subtraction_valid and (not addition_valid or np.random.randint(2)):
        # Check how many points can be removal in total
        total_max_removal = curr_num_points - true_min_points
        if count_limit:
            total_max_removal = min(total_max_removal, count_limit)
        total_max_removal = min(total_max_removal, (curr_width - true_min_width) // separation)
        
        # Check how many points can be removed from each side
        max_removal_from_min = 0
        if output_min_x <= max_first_x - separation:
            max_removal_from_min = min(total_max_removal, (max_first_x - output_min_x) // separation)
        max_removal_from_max = 0
        if output_max_x >= min_last_x + separation:
            max_removal_from_max = min(total_max_removal, (output_max_x - min_last_x) // separation)
        
        if max_removal_from_min and (not max_removal_from_max or np.random.randint(2)):
            point_diff = -1 if max_removal_from_min == 1 else -1*np.random.randint(1, max_removal_from_min + 1)
            return point_diff, True, None
        elif max_removal_from_max :
            point_diff = -1 if max_removal_from_max == 1 else -1*np.random.randint(1, max_removal_from_max + 1)
            return point_diff, False, None
        # print(f'did not change subtract: {max_removal_from_min}, {max_removal_from_min}')
        return 0, False, None
    
    elif addition_valid:
        # Check how many points can be added in total
        total_max_addition = true_max_points - curr_num_points
       
        if count_limit:
            total_max_addition= min(total_max_addition , count_limit)
        total_max_addition = min(total_max_addition, (curr_width - true_min_width) // separation)
        # print(f"total_max_addition: {total_max_addition} for curr_width {curr_width} and true_min_width {true_min_width} and separation {separation}")
        # Check how many points can be added from each side
        max_addition_to_min = 0
        if output_min_x - separation >= x_min:
            max_addition_to_min  = min(total_max_addition, (output_min_x - x_min) // separation)
        max_addition_to_max = 0
        if output_max_x + separation <= x_max:
            max_addition_to_max = min(total_max_addition, (x_max - output_max_x) // separation)

        if max_addition_to_min and (not max_addition_to_max or np.random.randint(2)):
            point_diff = 1 if max_addition_to_min == 1 else np.random.randint(1, max_addition_to_min + 1)
            return point_diff, True, None
        elif max_addition_to_max:
            point_diff = 1 if max_addition_to_max == 1 else np.random.randint(1, max_addition_to_max + 1)
            return point_diff, False, None
        return 0, True, None
    
    return 0, False, None

