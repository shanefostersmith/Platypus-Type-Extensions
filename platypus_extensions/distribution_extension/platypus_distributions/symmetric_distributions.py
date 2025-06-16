
import numpy as np
from collections.abc import Iterable
from math import ceil, floor
from platypus_extensions.utils import clip
from .symmetric_bijection import SymmetricBijection
from .point_bounds import PointBounds
from .monotonic_distributions import MonotonicDistributions
from ._distribution_tools import DistributionInfo

class SymmetricDistributions(MonotonicDistributions):
    """**Evolve and mutate a discrete set of symmetric distributions in the form of 1-D numpy arrays.**
    
    This is a subclass of MonotonicDistributions
    - SymmetricDistributions is compatibile with all of the same LocalVariators and LocalMutators
    - MonotonicDistributions and SymmetricBijection share the same decoded return type
    - The only difference is that SymmetricBijections are the input maps (instead of RealBijections).
    
    (See *SymmetricBijection* and *MonotonicDistributions*)
    """
    
    def __init__(
        self, 
        mappings: Iterable[SymmetricBijection],
        local_variator = None,
        local_mutator = None,
        ordinal_maps = False,
        ):
        """
        Args:
            mappings (Iterable[SymmetricBijection]): An iterable of SymmetricBijections with the inverse functions set.
            local_variator (LocalVariator, optional): Cannot be a LocalMutator, should have MonotonicDistributions or SymmetricDistributions registered in _supported_types. Defaults to None.
            local_mutator (LocalMutator, optional): A LocalMutator, should have MonotonicDistributions or SymmetricDistributions registered in _supported_types. Defaults to None.
            ordinal_maps (bool, optional): Indicates that the SymmetricBijection objects are inputted in some sort of order. Defaults to False.
            
        Raises:
            TypeError: If any of the input maps are not a SymmetricBijection
        """        
        
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
        
        # resolve discrepencies between full width bounds and half-width bounds
        full_min_separation, full_max_separation = bijection._full_separation_bounds
        true_min_separation = min(all_bounds.max_separation, max(full_min_separation, all_bounds.min_separation))
        true_max_separation = max(all_bounds.min_separation, min(full_max_separation, all_bounds.max_separation))
         
        if max_first_x == x_min and min_last_x == x_max: 
            width = x_max - x_min
            true_min_points = None
            true_max_points = None
            if all_bounds.min_points == all_bounds.max_points:
                true_min_points = all_bounds.min_points
                true_max_points = all_bounds.min_points
            else:
                true_max_points = floor(width / true_min_separation + 1.0)
                true_min_points = min(true_max_points, ceil(width / true_max_separation + 1.0))
                
            if true_min_points == true_max_points:
                output_separation = width / all_bounds.dtype(true_min_points - 1)
                output_points = true_min_points
            else:
                output_points = np.random.randint(true_min_points, true_max_points + 1)
                output_separation = width / all_bounds.dtype(output_points - 1)
                
            return  DistributionInfo(
                map_index=rand_function_idx,
                output_min_x= x_min,
                output_max_x= x_max,
                separation= output_separation,
                num_points= output_points 
            )
        
        output_min_x = bijection.center_x if bijection.right_side_provided else None
        output_max_x = None if bijection.right_side_provided else bijection.center_x
        output_points = None
        output_separation = None
        if true_min_separation == true_max_separation:
            output_separation = true_max_separation
            true_min_points, true_max_points = all_bounds.get_conditional_cardinality_bounds(output_separation)
            output_points = true_min_points if true_min_points == true_max_points else np.random.randint(true_min_points, true_max_points + 1)
        else:
            true_min_points = max(all_bounds.min_points, min(ceil(all_bounds.true_min_width / true_max_separation + 1.0), all_bounds.max_points))
            true_max_points = min(all_bounds.max_points, max(all_bounds.min_points, floor(all_bounds.true_max_width  / true_min_separation + 1.0)))
            output_points = true_min_points if true_min_points == true_max_points else np.random.randint(true_min_points, true_max_points + 1)
            new_min_separation, new_max_separation = all_bounds.get_conditional_separation_bounds(output_points)
            new_min_separation = max(true_min_separation, new_min_separation)
            new_max_separation = min(true_max_separation, new_max_separation)
            output_separation = np.random.uniform(new_min_separation, new_max_separation)
        
        new_width = all_bounds.dtype(output_points - 1) * output_separation
        assert not new_width > all_bounds.bound_width
        if bijection.right_side_provided:
            output_min_x = bijection.center_x 
            output_max_x = output_min_x + new_width
        else:
            output_max_x = bijection.center_x 
            output_min_x= output_max_x - new_width
        
        return DistributionInfo(
            map_index = rand_function_idx,
            output_min_x = output_min_x,
            output_max_x = output_max_x,
            num_points = output_points,
            separation =  output_separation,
        )
    
    def encode(self, value):
        
        y_distribution, map_idx = value
        bijection: SymmetricBijection = self.map_suite[map_idx]
        num_points = len(y_distribution)
        center_included = num_points % 2
        half_points = num_points // 2 + center_included
        output_min_x = None
        output_max_x = None
        separation = None
        if bijection.right_side_provided:
            first_idx = num_points // 2
            if center_included:
                output_min_x = bijection.center_x
            else:
                output_min_x = bijection.fixed_inverse_map(y_distribution[first_idx])
            second_x = bijection.fixed_inverse_map(y_distribution[first_idx+1])
            output_max_x = bijection.fixed_inverse_map(y_distribution[-1])
            separation = abs(second_x - output_min_x)
        else:
            first_idx = num_points // 2 if center_included else num_points // 2 - 1
            if center_included:
                output_max_x = bijection.center_x
            else:
                output_max_x = bijection.fixed_inverse_map(y_distribution[first_idx])
            second_x = bijection.fixed_inverse_map(y_distribution[first_idx-1])
            output_min_x = bijection.fixed_inverse_map(y_distribution[0])
            separation = abs(second_x - output_max_x)
            
        # resolve discrepencies between half and full bounds
        half_width = output_max_x - output_min_x
        if half_points > bijection.point_bounds.max_points:
            half_points = bijection.point_bounds.max_points
        elif half_points < bijection.point_bounds.min_points:
            half_points = bijection.point_bounds.min_points
        elif separation > bijection.point_bounds.max_separation: # decrease sep, increase points
            half_points = max(bijection.point_bounds.min_points, floor(half_width / bijection.point_bounds.max_separation + 1.0))
        elif separation < bijection.point_bounds.min_separation: # increase sep, decrease points
            half_points = min(bijection.point_bounds.max_points, ceil(half_width / bijection.point_bounds.min_separation + 1.0))
        else:
            return DistributionInfo(
                map_index = map_idx,
                num_points = half_points,
                separation = separation,
                output_min_x = output_min_x,
                output_max_x = output_max_x)
            
        separation = max(bijection.point_bounds.min_separation, min(bijection.point_bounds.max_separation, half_width / bijection.point_bounds.dtype(half_points - 1)))
        return DistributionInfo(
            map_index = map_idx,
            num_points = half_points,
            separation = separation,
            output_min_x = output_min_x,
            output_max_x = output_max_x)