import pytest
from platypus_extensions.distributions.monotonic_distributions import *
from ...distributions.tests.conftest import *

class TestFixedMapMonotonic:
    rtol = 1e-7
    atol= 1e-8
    
    def test_rand_and_decode(self, all_monotonic_distributions: MonotonicDistributions):
        distrib_info = all_monotonic_distributions.rand()
        bijection: RealBijection = all_monotonic_distributions.map_suite[distrib_info.map_index]
        assert distrib_info.output_min_x < distrib_info.output_max_x
        assert distrib_info.num_points <= bijection.point_bounds.max_points
        assert distrib_info.separation > 0
        if bijection.point_bounds.fixed_width is not None:
            assert np.isclose(distrib_info.output_max_x - distrib_info.output_min_x, bijection.point_bounds.fixed_width, self.rtol, self.atol)
        
        decoded_distribution, _ = all_monotonic_distributions.decode(distrib_info)
        unique_elems = np.unique(decoded_distribution)
        assert len(unique_elems) == len(decoded_distribution)
        if all_monotonic_distributions.sort_ascending:
            assert np.all(decoded_distribution[:-1] < decoded_distribution[1:])
        else:
            assert np.all(decoded_distribution[:-1] > decoded_distribution[1:])
   
    def test_pcx(self, distribution_pcx: MonotonicDistributions, nsolutions_crossover, request): #deepcopy_parents,):
        nparents, offspring = nsolutions_crossover
        parent_solutions, offspring_sol, copy_indices = create_one_var_solutions(distribution_pcx, nparents, offspring, deepcopy=False)
        orig_parent_vars = [copy.deepcopy(sol.variables[0]) for sol in parent_solutions]
        orig_offspring_vars = [copy.deepcopy(sol.variables[0]) for sol in offspring_sol]
        distribution_pcx.local_variator.evolve(distribution_pcx, parent_solutions, offspring_sol, variable_index=0, copy_indices=copy_indices)
        
        for i, o in enumerate(offspring_sol):
            distribution_info = o.variables[0]
            orig_distribution_info = orig_offspring_vars[i]
            map_idx = distribution_info.map_index
            assert map_idx == orig_distribution_info.map_index
            width = distribution_info.output_max_x - distribution_info.output_min_x 
            assert width > 0
            
            bijection: RealBijection = distribution_pcx.map_suite[map_idx]
            width2 = distribution_info.separation * bijection.dtype(distribution_info.num_points - 1)
            assert np.isclose(width, width2, self.rtol, self.atol), f"width = {width}, width2 = {width2}"
            
            if  bijection.point_bounds.fixed_width:
                assert np.isclose(width, bijection.point_bounds.fixed_width, self.rtol, self.rtol), (
                f"true_min: {bijection.point_bounds.true_min_width}, true_max: {bijection.point_bounds.true_max_width}\n",
                f"original width: {distribution_info.output_max_x - distribution_info.output_min_x} ")
            
            assert bijection.point_bounds.lower_bound < distribution_info.output_min_x or (
                np.isclose(bijection.point_bounds.lower_bound, distribution_info.output_min_x, self.rtol, self.atol)
            )
            assert bijection.point_bounds.upper_bound > distribution_info.output_max_x or (
                np.isclose(bijection.point_bounds.upper_bound, distribution_info.output_max_x, self.rtol, self.atol)
            )
            assert bijection.point_bounds.min_points <= distribution_info.num_points <= bijection.point_bounds.max_points
            assert distribution_info.separation >= bijection.point_bounds.min_separation or (
                np.isclose(bijection.point_bounds.min_separation , distribution_info.separation, self.rtol, self.atol)
            )
            
            decoded_distribution, _ = distribution_pcx.decode(distribution_info)
            unique_elems = np.unique(decoded_distribution)
            assert len(unique_elems) == len(decoded_distribution), (
                f"spacing_min {np.spacing(distribution_info.output_min_x)} spacing max: {np.spacing(distribution_info.output_max_x)}, distribution_info: {distribution_info!r} ")
            if distribution_pcx.sort_ascending:
                assert np.all(decoded_distribution[:-1] < decoded_distribution[1:])
            else:
                assert np.all(decoded_distribution[:-1] > decoded_distribution[1:])
        
        for i, p in enumerate(parent_solutions):
            assert orig_parent_vars[i] == p.variables[0]
    
    def test_map_crossover(self, distribution_map_crossover: MonotonicDistributions, nsolutions_crossover):
        nparents, offspring = nsolutions_crossover
        parent_solutions, offspring_sol, copy_indices = create_one_var_solutions(distribution_map_crossover, nparents, offspring, deepcopy=True)
        orig_parent_vars = [copy.deepcopy(sol.variables[0]) for sol in parent_solutions]
        # orig_offspring_vars = [copy.deepcopy(sol.variables[0]) for sol in offspring_sol]
        distribution_map_crossover.local_variator.equal_map_crossover = True
        distribution_map_crossover.local_variator.evolve(distribution_map_crossover, parent_solutions, offspring_sol, variable_index=0, copy_indices=copy_indices)
        
        for i, o in enumerate(offspring_sol):
            distribution_info = o.variables[0]
            width = distribution_info.output_max_x - distribution_info.output_min_x 
            assert width > 0
            
            bijection: RealBijection = distribution_map_crossover.map_suite[distribution_info.map_index]
            width2 = distribution_info.separation * bijection.dtype(distribution_info.num_points - 1)
            assert np.isclose(width, width2, self.rtol, self.atol), f"width = {width}, width2 = {width2}"
            
            if  bijection.point_bounds.fixed_width:
                assert np.isclose(width, bijection.point_bounds.fixed_width, self.rtol, self.rtol), (
                f"true_min: {bijection.point_bounds.true_min_width}, true_max: {bijection.point_bounds.true_max_width}\n",
                f"original width: {distribution_info.output_max_x - distribution_info.output_min_x} ")
            
            assert bijection.point_bounds.lower_bound < distribution_info.output_min_x or (
                np.isclose(bijection.point_bounds.lower_bound, distribution_info.output_min_x, self.rtol, self.atol)
            )
            assert bijection.point_bounds.upper_bound > distribution_info.output_max_x or (
                np.isclose(bijection.point_bounds.upper_bound, distribution_info.output_max_x, self.rtol, self.atol)
            )
            assert bijection.point_bounds.min_points <= distribution_info.num_points <= bijection.point_bounds.max_points
            assert distribution_info.separation >= bijection.point_bounds.min_separation or (
                np.isclose(bijection.point_bounds.min_separation , distribution_info.separation, self.rtol, self.atol)
            )
            
            decoded_distribution, _ = distribution_map_crossover.decode(distribution_info)
            unique_elems = np.unique(decoded_distribution)
            assert len(unique_elems) == len(decoded_distribution), (
                f"spacing_min {np.spacing(distribution_info.output_min_x)} spacing max: {np.spacing(distribution_info.output_max_x)}, distribution_info: {distribution_info!r} ")
            if distribution_map_crossover.sort_ascending:
                assert np.all(decoded_distribution[:-1] < decoded_distribution[1:])
            else:
                assert np.all(decoded_distribution[:-1] > decoded_distribution[1:])
        
        for i, p in enumerate(parent_solutions):
            assert orig_parent_vars[i] == p.variables[0]
    
    def test_map_conversion(self, distribution_map_mutation: MonotonicDistributions):
        _, offspring_sol, _ = create_one_var_solutions(distribution_map_mutation, 1, 1)
        orig_map = offspring_sol.variables[0].map_index
        
        distribution_map_mutation.local_variator.mutate(distribution_map_mutation, offspring_sol, variable_index=0)
        assert offspring_sol.evaluated == False if offspring_sol.variables[0].map_index != orig_map else True

        distribution_info = offspring_sol.variables[0]
        width = distribution_info.output_max_x - distribution_info.output_min_x 
        assert width > 0
        
        bijection: RealBijection = distribution_map_mutation.map_suite[distribution_info.map_index]
        width2 = distribution_info.separation * bijection.dtype(distribution_info.num_points - 1)
        assert np.isclose(width, width2, self.rtol, self.atol), f"width = {width}, width2 = {width2}"
        
        if  bijection.point_bounds.fixed_width:
            assert np.isclose(width, bijection.point_bounds.fixed_width, self.rtol, self.rtol), (
            f"true_min: {bijection.point_bounds.true_min_width}, true_max: {bijection.point_bounds.true_max_width}\n",
            f"original width: {distribution_info.output_max_x - distribution_info.output_min_x} ")
        
        assert bijection.point_bounds.lower_bound < distribution_info.output_min_x or (
            np.isclose(bijection.point_bounds.lower_bound, distribution_info.output_min_x, self.rtol, self.atol)
        )
        assert bijection.point_bounds.upper_bound > distribution_info.output_max_x or (
            np.isclose(bijection.point_bounds.upper_bound, distribution_info.output_max_x, self.rtol, self.atol)
        )
        assert bijection.point_bounds.min_points <= distribution_info.num_points <= bijection.point_bounds.max_points
        assert distribution_info.separation >= bijection.point_bounds.min_separation or (
            np.isclose(bijection.point_bounds.min_separation , distribution_info.separation, self.rtol, self.atol)
        )
    
    def test_other_mutation(self, distribution_bound_mutation, request):
        mutation_type = request.getfixturevalue("distribution_mutator")
        # print(f"mutation_type: {mutation_type}")
        _, offspring_sol, _ = create_one_var_solutions(distribution_bound_mutation, 1, 1)
        orig_offspring_info = copy.deepcopy(offspring_sol.variables[0])
        bijection: RealBijection = distribution_bound_mutation.map_suite[orig_offspring_info.map_index]

        if mutation_type == 'points':
            distribution_bound_mutation.local_variator.mutate(distribution_bound_mutation, offspring_sol, variable_index=0)
            new_offspring_info = offspring_sol.variables[0]
            if bijection.point_bounds.min_points == bijection.point_bounds.max_points:
                distribution_bound_mutation.local_variator.mutate(distribution_bound_mutation, offspring_sol, variable_index=0)
                new_offspring_info = offspring_sol.variables[0]
                assert offspring_sol.evaluated and bijection.point_bounds.min_points == new_offspring_info.num_points
            else:
                count_limits = (None, 1, 2, 10)
                for lim in count_limits:
                    if lim is not None and lim > bijection.point_bounds.max_points - bijection.point_bounds.min_points:
                        break
                    
                    orig_offspring_info = copy.deepcopy(offspring_sol.variables[0])
                    distribution_bound_mutation.local_variator.mutation_count_limit = lim
                    distribution_bound_mutation.local_variator.mutate(distribution_bound_mutation, offspring_sol, variable_index=0)
                    new_offspring_info = offspring_sol.variables[0]
                
                    assert bijection.point_bounds.min_points <= new_offspring_info.num_points <= bijection.point_bounds.max_points
                    assert offspring_sol.evaluated == False if new_offspring_info.num_points != orig_offspring_info.num_points else True
                    
                    point_dist = new_offspring_info.separation * bijection.dtype(new_offspring_info.num_points - 1)
                    if bijection.point_bounds.fixed_width or bijection.point_bounds.true_min_width == bijection.point_bounds.true_max_width:
                        assert np.isclose(
                            orig_offspring_info.output_max_x - orig_offspring_info.output_min_x,
                            point_dist,
                            self.rtol, self.atol
                        )
                    else:
                        assert point_dist <= np.nextafter(bijection.point_bounds.bound_width, np.inf)
                        assert point_dist >= np.nextafter(bijection.point_bounds.true_min_width, -np.inf)
                        
                        assert bijection.point_bounds.max_first_point > new_offspring_info.output_min_x or (
                            np.isclose(bijection.point_bounds.max_first_point, new_offspring_info.output_min_x, self.rtol, self.atol)
                        )
                        assert bijection.point_bounds.min_last_point < new_offspring_info.output_max_x or (
                            np.isclose(bijection.point_bounds.min_last_point, new_offspring_info.output_max_x, self.rtol, self.atol)
                        )
                        assert bijection.point_bounds.lower_bound < new_offspring_info.output_min_x or (
                            np.isclose(bijection.point_bounds.lower_bound, new_offspring_info.output_min_x, self.rtol, self.atol)
                        )
                        assert bijection.point_bounds.upper_bound > new_offspring_info.output_max_x or (
                            np.isclose(bijection.point_bounds.upper_bound, new_offspring_info.output_max_x, self.rtol, self.atol)
                        )
                    
                    if lim is not None:
                        assert abs(new_offspring_info.num_points - orig_offspring_info.num_points) <= lim
                    else:
                        assert abs(new_offspring_info.num_points - orig_offspring_info.num_points) <= bijection.point_bounds.max_points - bijection.point_bounds.min_points
                        
        elif mutation_type == 'shift':
            distribution_bound_mutation.local_variator.mutate(distribution_bound_mutation, offspring_sol, variable_index=0)
            new_offspring_info = offspring_sol.variables[0]
            assert bijection.point_bounds.lower_bound < new_offspring_info.output_min_x or (
                np.isclose(bijection.point_bounds.lower_bound, new_offspring_info.output_min_x, self.rtol, self.atol)
            )
            assert bijection.point_bounds.upper_bound > new_offspring_info.output_max_x or (
                np.isclose(bijection.point_bounds.upper_bound, new_offspring_info.output_max_x, self.rtol, self.atol)
            )
            assert bijection.point_bounds.max_first_point > new_offspring_info.output_min_x or (
                np.isclose(bijection.point_bounds.max_first_point, new_offspring_info.output_min_x, self.rtol, self.atol)
            )
            assert bijection.point_bounds.min_last_point < new_offspring_info.output_max_x or (
                np.isclose(bijection.point_bounds.min_last_point, new_offspring_info.output_max_x, self.rtol, self.atol)
            )
            
        
        elif mutation_type == 'separation' and bijection.point_bounds.min_separation != bijection.point_bounds.max_separation:
            # print(f"B: {bijection.point_bounds}")
            # print(f"previous_separation: {orig_offspring_info.separation}")
            # print(f"previous min / max: ({orig_offspring_info.output_min_x}, {orig_offspring_info.output_max_x})")
            distribution_bound_mutation.local_variator.mutate(distribution_bound_mutation, offspring_sol, variable_index=0)
            new_offspring_info = offspring_sol.variables[0]
            # print(f"new_separation: {new_offspring_info.separation}")
            # print(f"new min / max: ({new_offspring_info.output_min_x}, {new_offspring_info.output_max_x})\n")
            
            point_dist = new_offspring_info.separation * bijection.dtype(new_offspring_info.num_points - 1)
            assert point_dist <= np.nextafter(bijection.point_bounds.bound_width, np.inf)
            assert point_dist >= np.nextafter(bijection.point_bounds.true_min_width, -np.inf)
            
            assert bijection.point_bounds.min_separation <= new_offspring_info.separation or (
                np.isclose(bijection.point_bounds.min_separation, new_offspring_info.separation, self.rtol, self.atol)
            )
            assert bijection.point_bounds.max_separation >= new_offspring_info.separation or (
                np.isclose(bijection.point_bounds.max_separation, new_offspring_info.separation, self.rtol, self.atol)
            )
            assert bijection.point_bounds.lower_bound < new_offspring_info.output_min_x or (
                np.isclose(bijection.point_bounds.lower_bound, new_offspring_info.output_min_x, self.rtol, self.atol)
            )
            assert bijection.point_bounds.upper_bound > new_offspring_info.output_max_x or (
                np.isclose(bijection.point_bounds.upper_bound, new_offspring_info.output_max_x, self.rtol, self.atol)
            )
            assert bijection.point_bounds.max_first_point > new_offspring_info.output_min_x or (
                np.isclose(bijection.point_bounds.max_first_point, new_offspring_info.output_min_x, self.rtol, self.atol)
            )
            assert bijection.point_bounds.min_last_point < new_offspring_info.output_max_x or (
                np.isclose(bijection.point_bounds.min_last_point, new_offspring_info.output_max_x, self.rtol, self.atol)
            )

            
            
            