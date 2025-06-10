
import numpy as np
from collections.abc import Iterable
from .symmetric_bijection import SymmetricBijection
from .point_bounds import PointBounds
from .monotonic_distributions import MonotonicDistributions
from ._distribution_tools import DistributionInfo
from ..utils import clip
from math import ceil, floor

class SymmetricDistributions(MonotonicDistributions):
    """(in progress)"""
    
    def __init__(
        self, 
        mappings: Iterable[SymmetricBijection],
        local_variator = None,
        local_mutator = None,
        ordinal_maps = False,
        ):
        
        for i, map in enumerate(mappings):
            if not isinstance(map, SymmetricBijection):
                raise TypeError(f"Input map {i} is not a SymmetricBijection object")
            
        super(SymmetricDistributions, self).__init__(
            mappings,
            local_variator,
            local_mutator,
            ordinal_maps,
            sort_ascending=True,
        )
    
    def rand(self):
        rand_function_idx = 0 if self.num_functions == 1 else np.random.randint(0,self.num_functions)
        bijection: SymmetricBijection = self.map_suite[rand_function_idx] 
        all_bounds = bijection.point_bounds
        
        x_min, max_first_x = all_bounds.first_point_bounds
        min_last_x, x_max = all_bounds.last_point_bounds
        
        
        if max_first_x == x_min and min_last_x == x_max: # have to choose separation after points
            width = x_max - x_min
            true_min_points, true_max_points = all_bounds.get_conditional_cardinality_with_width(width)
            output_points = true_min_points if true_min_points == true_max_points else np.random.int(true_min_points, true_max_points)
            separation = width / all_bounds.dtype(output_points - 1)
            output_min_x = x_min
            output_max_x = x_max
            return  DistributionInfo(
                map_index = rand_function_idx,
                num_points = output_points,
                separation =  separation ,
                output_min_x = output_min_x,
                output_max_x = output_max_x
            )
        output_min_x = None
        output_max_x = None
        output_points = None
            
        #TODO: Find bounds
        
        return DistributionInfo(
            map_index = rand_function_idx,
            num_points = output_points,
            separation =  separation ,
            output_min_x = output_min_x,
            output_max_x = output_max_x
        )
    
    def encode(self, value):
        
        y_distribution, map_idx = value
        bijection: SymmetricBijection = self.map_suite[map_idx]
        
        num_points = len(y_distribution)
        center_included = num_points % 2
        half_points = num_points // 2 + center_included
        assert half_points >= bijection.point_bounds.min_points, f"Too few points in current distribution ({num_points}) , minimum is {bijection.point_bounds.min_points})"
        first_y = None
        last_y = None
        if bijection.right_side_provided:
            first_y = y_distribution[-half_points] #TODO CHECK
            last_y = y_distribution[-1]
        else:
            first_y = y_distribution[0]
            last_y = y_distribution[half_points - 1]
            
        first_idx_x = bijection.fixed_inverse_map(first_y)
        last_idx_x = bijection.fixed_inverse_map(last_y) 
        assert not (first_idx_x is None or last_idx_x is None), "The inverse function returned None values"
        assert first_idx_x != last_idx_x, "The first 'x' value in current distribution equals the last 'x' value"
        
        output_min_x = min(first_idx_x, last_idx_x)
        output_max_x = max(first_idx_x, last_idx_x)
        separation = (output_max_x - output_min_x) / bijection._return_type(half_points - 1)
        
        distribution_info = DistributionInfo(
            map_index = map_idx,
            num_points = num_points,
            separation = separation,
            output_min_x = output_min_x,
            output_max_x = output_max_x)
        
        return distribution_info
    
# def _count_mutation(self, 
#                     bijection: SymmetricBijection,
#                     output_min_x, output_max_x, 
#                     curr_num_points):
#     """
#     Remove or add points from distribution. 
    
#     No change to width (unlike monotonic case)

#     Returns:
#         tuple(int, bool): (point_diff, diff_at_min_x, separation_change)
#             i.e. (number points added/removed, if adding or removing points from start x or last x, if changed separation instead of width)
#     """        
    
    
#     all_x_bounds: PointBounds = bijection.get_all_x_bounds()
#     curr_width = output_max_x - output_min_x

#     true_min_points, true_max_points = all_x_bounds.get_conditional_points_with_width(curr_width)
#     max_addition = max(0, true_max_points - curr_num_points)
#     max_subtraction = max(0,curr_num_points - true_min_points)
#     if self.sample_count_mutation_limit:
#         max_addition = min(self.sample_count_mutation_limit, max_addition)
#         max_subtraction= min(self.sample_count_mutation_limit, max_subtraction)
        
#     point_diff = 0
#     if max_subtraction and (not max_addition or np.random.randint(2)):
#         point_diff = 1 if max_subtraction == 1 else np.random.randint(1, max_subtraction + 1)
#         return -point_diff, curr_width  / all_x_bounds.dtype(curr_num_points - point_diff - 1)
#     elif max_addition:
#         point_diff = 1 if max_addition == 1 else np.random.randint(1, max_addition + 1)
#         return point_diff, curr_width  / all_x_bounds.dtype(curr_num_points + point_diff - 1)
#     return 0, None
    
    
    

    