import numpy as np
from ..utils import _min_max_norm_convert
from ._bounds_tools import BoundsViewMixin
from ._distribution_tools import DistributionInfo, ordered_y_bounds
from .real_bijection import RealBijection
from collections import namedtuple

NormalizedOutput = namedtuple('NormalizedOutput', ['start', 'end', 'points'], defaults = (None, None, None))

def _convert_new_start_y(
        bijection: RealBijection,
        new_y, 
        x_min, x_max, 
        max_first_x, min_last_x,
        y_min, max_first_y
    ):
        """
        Called after evolving a minimum y value, associated x value in bound limits
            May return a new start x or new end x
        
        Returns `(new_start_x, new_end_x)`
        - One of `new_start_x` or `new_end_x` will be None
        """
        assert x_min is not None and x_max is not None
        decreasing = bijection.direction
        new_x = None
        new_y = max(y_min, min(new_y ,max_first_y))
        
        if new_y <= y_min:
            new_x = x_max if decreasing else x_min 
        elif new_y >= max_first_y:
            new_x = min_last_x if decreasing else max_first_x  
        else:
            new_x = bijection.fixed_inverse_map(new_y)

        new_end_x = new_x if decreasing else None
        new_start_x = None if new_end_x is not None else new_x
        return new_start_x, new_end_x

def _convert_new_end_y(
        bijection:RealBijection,
        new_end_y, 
        x_min, x_max, 
        max_first_x, min_last_x,
        y_max, min_last_y
    ):
        """
        Called after evolving a maximum y value, find associated x value in bounds
            May return a new start x or new end x
        
        Returns `new_x`
        - May be `x_min` (if decreasing function) or `x_max` (if increasing function)
        """
        decreasing = bijection.direction
        new_x = None
        new_end_y = max(min_last_y, min(new_end_y, y_max))
        
        if new_end_y == y_max:
            new_x = x_min if decreasing else x_max
        elif new_end_y == min_last_y:
            new_x = max_first_x if decreasing else min_last_x
        else:
            new_x = bijection.fixed_inverse_map(new_end_y)
        
        return new_x

def apply_y_bound_pcx(
    new_offspring_vars: NormalizedOutput,
    offspring_info: DistributionInfo,
    bijection: RealBijection,
    global_min_points: int,
    global_max_points: int,
    global_min_y: float,
    global_max_y: float
):
    """Updates offspring distribution with normalized/evolved "y" values. 
    
    `new_offspring_vars` may contain 1, 2 or 3 variables
    - First var always start y
    - Second var is points if 'evolve_points' is True. Otherwise, it is an end y
    - If there's a third var, is is an end y 
    
    (If fixed_width, no end y)

    Args:
        new_offspring_vars (np.ndarray): _description_
        offspring_info (DistributionInfo): _description_
        evolve_points (bool): _description_
        fixed_width (bool): _description_
        global_min_points (int): _description_
        global_max_points (int): _description_
        global_y_width (float): _description_
    """    
    x_bounds = bijection.point_bounds
    # print(f"INNER MAP IDX: {offspring_info.map_index}")
    # print(f"INNER Y_BASED bounds: {x_bounds!r}")
    y_min, y_max, max_first_y, min_last_y = ordered_y_bounds(bijection)
    new_x_width = None
    true_min_width = x_bounds.true_min_width
    true_max_width = x_bounds.true_max_width
    if true_min_width >= true_max_width:
        new_x_width = true_max_width

    new_start_y = _min_max_norm_convert(global_min_y, global_max_y, new_offspring_vars.start, to_norm = False)
    new_start_x, new_end_x = _convert_new_start_y(
        bijection, new_start_y, 
        x_bounds.lower_bound, x_bounds.upper_bound, 
        x_bounds.max_first_point, x_bounds.min_last_point,
        y_min, max_first_y
    )
    
    if new_x_width is not None: # Fixed width
        new_start_x = max(x_bounds.lower_bound, new_start_x if new_start_x is not None else new_end_x - new_x_width)
    else: # Convert end_y, find x_width
        new_end_y = _min_max_norm_convert(global_min_y, global_max_y, new_offspring_vars.end, to_norm = False)
        other_x = _convert_new_end_y(
            bijection, new_end_y, 
            x_bounds.lower_bound, x_bounds.upper_bound, 
            x_bounds.max_first_point, x_bounds.min_last_point,
            y_max, min_last_y
        )
        if new_start_x is not None:
            new_end_x = other_x
        else:
            new_start_x = other_x

        new_x_width = x_bounds.dtype(max(true_min_width, min((new_end_x - new_start_x), true_max_width)))
        
    new_end_x = x_bounds.dtype(new_start_x) + new_x_width
    new_points = None
    if new_offspring_vars.points is not None and x_bounds.min_points != x_bounds.max_points:
        new_flt_points = _min_max_norm_convert(np.float32(global_min_points), np.float32(global_max_points), new_offspring_vars.points, to_norm = False)
        true_min_points, true_max_points = x_bounds.get_conditional_cardinality_with_width(new_x_width)
        new_points = int(max(true_min_points, min(new_flt_points, true_max_points)))
    else:
        new_points = x_bounds.min_points
    
    offspring_info.num_points = new_points
    offspring_info.output_min_x = new_start_x
    offspring_info.output_max_x = new_end_x
    offspring_info.separation = new_x_width / x_bounds.dtype(new_points - 1)
  
def apply_x_bound_pcx(
    new_offspring_vars: NormalizedOutput,
    offspring_info: DistributionInfo,
    x_bounds: BoundsViewMixin,
    global_min_points: int,
    global_max_points: int,
):
    """Updates offspring distribution with normalized/evolved "x" values. 
    
    `new_offspring_vars` always has a numeric 'start', 
    
    'points' and 'end' may be None
    
    """    
    new_x_width = None
    true_min_width = x_bounds.true_min_width
    true_max_width = x_bounds.true_max_width
    if true_min_width >= true_max_width:
        new_x_width = true_max_width
    else:
        assert x_bounds.fixed_width is None
    x_min, max_first_x = x_bounds.first_point_bounds
    min_last_x, x_max = x_bounds.last_point_bounds
    new_max_first = min(max_first_x, x_max - true_min_width)
    
    new_start_x = max(x_min, min(new_max_first, _min_max_norm_convert(x_min, x_max, new_offspring_vars.start, False)))
    new_end_x = None
    if new_x_width is None: 
        new_min_last = max(min_last_x, new_start_x + true_min_width)
        assert new_offspring_vars.end is not None
        new_end_x = max(new_min_last, min(x_max, _min_max_norm_convert(x_min, x_max, new_offspring_vars.end, False)))
        new_x_width = x_bounds.dtype(min(true_max_width, new_end_x - new_start_x))
    
    new_end_x = x_bounds.dtype(new_start_x) + new_x_width
    new_points = None
    if new_offspring_vars.points is not None and x_bounds.min_points != x_bounds.max_points:
        new_flt_points = _min_max_norm_convert(np.float32(global_min_points), np.float32(global_max_points), new_offspring_vars.points, to_norm = False)
        true_min_points, true_max_points = x_bounds.get_conditional_cardinality_with_width(new_x_width)
        new_points = int(max(true_min_points, min(new_flt_points, true_max_points)))
    else:
        new_points = x_bounds.min_points

    offspring_info.num_points = new_points
    offspring_info.output_min_x = new_start_x
    offspring_info.output_max_x = new_end_x
    offspring_info.separation = new_x_width / x_bounds.dtype(new_points - 1)

#EXPERIMENTAL DE
# def differential_evolution(
#         self, 
#         parents: Sequence[Solution],
#         variable_index: int,
#         parent_distribution_info: list, 
#         offspring_map_indices: list,
#         step_size):
#         """Parent info / offspring info elements in format
#         [distribution_info (tuple), (min_width, max_width)]
        
#         distribution_info (tuple) in format:
        
#         (output_min_x, output_max_x, 
#         output_min_y , output_max_y,
#         width, num_points, separation)
        
#         (see get_distribution_info())

#         Assumes number of parents = 4
#         """
        
#         assert(len(offspring_map_indices) == 1 and len(parent_distribution_info) == 4), f"If differential evolution, should be 1 offspring and 4 parents (got {len(offspring_map_indices)} offspring and {len(parent_distribution_info)} parents)"
#         curr_map = offspring_map_indices[0]
#         curr_bijection: RealBijection = self.map_suite[curr_map]
#         curr_x_bounds: PointBounds = curr_bijection.get_all_x_bounds()
#         true_min_width, true_max_width = self.get_true_width_bounds(curr_bijection)
#         normalized_parent_info = []

 
#         for i in range(1,4):
#             info_list = parent_distribution_info[i]
#             _, parent_map = parents[i].variables[variable_index]
#             parent_bijection = self.map_suite[parent_map]
#             normalized_info = self.normalize_distribution_info(info_list[0], parent_bijection)
#             normalized_parent_info.append(normalized_info)
        
#         evolve_start_x = not curr_x_bounds._bounds[0] == curr_x_bounds._max_first_point
#         #TODO (add case for fixed ends)
            
#         # Evolve start or end x
#         new_start_x = None
#         new_end_x = None
#         new_width = curr_x_bounds._fixed_width
#         x_min, max_first_x = curr_x_bounds.get_first_point_bounds(True)
#         min_last_x, x_max = curr_x_bounds.get_last_point_bounds(True)
#         if not self.y_based_crossover:
#             parent_out_x = [parent_info[0] for parent_info in normalized_parent_info] if evolve_start_x else [parent_info[1] for parent_info in normalized_parent_info] 
#             norm_evolved_start = differential_evolve(0.0, 1.0, parent_out_x[0], parent_out_x[1], parent_out_x[2], step_size, False)
#             new_out_x = _min_max_norm_convert(x_min, x_max, norm_evolved_start, False)
#             if evolve_start_x:
#                 new_start_x = max(x_min, min(new_out_x, max_first_x))
#             else:
#                 new_end_x = max(min_last_x, min(new_out_x, x_max))
            
#             # Evolve other x
#             if new_width is None:
#                 parent_other_x = [parent_info[1] for parent_info in normalized_parent_info] if evolve_start_x else [parent_info[0] for parent_info in normalized_parent_info] 
#                 norm_evolved_end = differential_evolve(0.0, 1.0, parent_other_x[0], parent_other_x[1], parent_other_x[2], step_size, False)
#                 new_other_x = _min_max_norm_convert(x_min, x_max, norm_evolved_end, False)
#                 if evolve_start_x:
#                     min_last_x = max(min_last_x, new_start_x + true_min_width)
#                     new_end_x= max(min_last_x, min(new_other_x, x_max))
#                 else:
#                     max_first_x = min(max_first_x, new_end_x - true_min_width)
#                     new_start_x = max(x_min, min(new_other_x, max_first_x))
#                 new_width = min(true_max_width, new_end_x - new_start_x)
#             elif new_start_x is None:
#                 new_start_x = new_end_x - new_width
                
#         # Evolve start or end y 
#         else: 
#             decreasing = curr_bijection.get_direction(False)
#             evolve_start_y = decreasing != evolve_start_x
#             parent_out_y = [parent_info[2] for parent_info in normalized_parent_info] if evolve_start_y else [parent_info[3] for parent_info in normalized_parent_info]
#             norm_evolved_y = differential_evolve(0.0, 1.0, parent_out_y[0], parent_out_y[1], parent_out_y [2], step_size, False)
#             y_min, y_max, max_first_y, min_last_y = self.ordered_y_bounds(curr_bijection)
#             new_start_y = None
#             new_end_y = None
#             if evolve_start_y:
#                 new_start_y = _min_max_norm_convert(self.global_min_y, self.global_max_y, norm_evolved_y, False)
#                 new_start_x, new_end_x, new_width = self._convert_new_start_y(
#                     curr_bijection, new_start_y, 
#                     x_min, x_max, max_first_x, min_last_x,
#                     y_min, max_first_y, true_min_width)
#             else:
#                 new_end_y = _min_max_norm_convert(self.global_min_y, self.global_max_y, norm_evolved_y, False)
#                 new_start_x, new_end_x, new_width = self._convert_new_end_y(
#                     curr_bijection, new_end_y, 
#                     x_min, x_max, max_first_x, min_last_x,
#                     y_max, min_last_y, true_min_width)
                
#             # Evolve y width
#             if new_width is not None:
#                 new_start_x = new_start_x if new_start_x is not None else new_end_x - new_width
#             else: 
#                 parent_y_width = [abs(parent_info[3] - parent_info[2]) for parent_info in normalized_parent_info]
#                 norm_evolved_width = differential_evolve(0.0, 1.0, parent_y_width[0], parent_y_width[1], parent_y_width[2], step_size, False)
#                 evolved_y_width = (self.global_max_y - self.global_min_y) * norm_evolved_width

#                 new_start_x, new_width = self.find_new_start_x_with_y_width(
#                     curr_bijection,
#                     evolved_y_width,
#                     new_start_x, new_end_x,
#                     new_start_y, new_end_y,
#                     x_min, x_max, max_first_x, min_last_x,
#                     y_min, y_max, max_first_y, min_last_y,
#                     true_min_width, true_max_width
#                 )
                      
#         # Evolve num points + separation
#         true_min_points, true_max_points = curr_x_bounds.get_conditional_points_with_width(new_width)
#         new_num_points = None
#         if true_min_points == true_max_points:
#             new_num_points = true_min_points
#         else:
#             parent_num_points = [parent_info[-2] for parent_info in normalized_parent_info]
#             norm_evolved_points = differential_evolve(0.0, 1.0, parent_num_points[0], parent_num_points[1], parent_num_points[2], step_size, False)
#             new_num_points = _min_max_norm_convert(np.float64(self.global_min_points), np.float64(self.global_max_points), norm_evolved_points, False)
#             new_num_points = max(true_min_points, min(true_max_points, new_num_points))
#         new_separation = new_width / curr_x_bounds.dtype(new_num_points - 1)
        
#         return [(curr_map, new_start_x, new_num_points, new_separation)]
    