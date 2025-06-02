import pytest
from custom_types.distributions.monotonic_distributions import *
from tests.test_distributions.conftest import *

class TestFixedMapMonotonic:
    
    def test_rand_and_decode(self, all_monotonic_distributions: MonotonicDistributions, request):
        distrib_info = all_monotonic_distributions.rand()
        bijection: RealBijection = all_monotonic_distributions.map_suite[distrib_info.map_index]
        fixed_type, fixed_value = request.getfixturevalue("combined_mono_distribution_options")
        # print(f"BOUND_TYPE: {fixed_type}, Value: {fixed_value}")
        assert distrib_info.output_min_x < distrib_info.output_max_x
        assert distrib_info.num_points <= bijection.point_bounds.max_points
        assert distrib_info.separation > 0
        
        decoded_distribution, _ = all_monotonic_distributions.decode(distrib_info)
        unique_elems = np.unique(decoded_distribution)
        assert len(unique_elems) == len(decoded_distribution)
        if all_monotonic_distributions.sort_ascending:
            assert np.all(decoded_distribution[:-1] < decoded_distribution[1:])
        else:
            assert np.all(decoded_distribution[:-1] > decoded_distribution[1:])
   
    def test_pcx(self, distribution_pcx: MonotonicDistributions, nsolutions_crossover): #deepcopy_parents,):
        nparents, offspring = nsolutions_crossover
        parent_solutions, offspring_solutions, copy_indices = create_one_var_solutions(distribution_pcx, nparents, offspring, deepcopy=True)
        
        orig_parent_vars = [np.copy(sol.variables[0]) for sol in parent_solutions]
        distribution_pcx.local_variator.evolve(distribution_pcx, parent_solutions, offspring_solutions, variable_index=0, copy_indices=copy_indices)
        