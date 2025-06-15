import pytest
import numpy as np
from platypus_extensions.bins_and_sets.bins_and_sets import *
from tests.conftest import create_one_var_solutions, deepcopy_parents
from tests.test_bins_sets.conftest import *

def test_set_partition_rand(set_partition_start, set_partition_dim):
    equal_start = set_partition_start == 'equal'
    total_features, subset_size, bins = set_partition_dim
    set_parition = SetPartition(total_features, subset_size, bins, equal_start)
    directory = set_parition.rand()
    # print(f"directory {directory}")
    
    assert len(directory) == total_features
    actives = directory[directory >= 0]
    assert len(actives) == subset_size
    assert np.count_nonzero(directory >= bins) == 0

def test_set_partition_crossover(set_partition_with_crossover, nsolutions_crossover, request):
    nparents, offspring = nsolutions_crossover
    total_features, subset_size, bins = request.getfixturevalue("set_partition_dim")
    variator_name = request.getfixturevalue("set_partition_variator")
    parent_sol, offspring_sol, copy_indices = create_one_var_solutions(set_partition_with_crossover, nparents, offspring)

    orig_parent_vars = [np.copy(sol.variables[0]) for sol in parent_sol]
    orig_offspring_vars = None if variator_name != 'active_swap' else [np.copy(sol.variables[0]) for sol in offspring_sol]

    set_partition_with_crossover.local_variator.evolve(set_partition_with_crossover, parent_sol, offspring_sol, variable_index=0, copy_indices = copy_indices)
    for i, o in enumerate(offspring_sol):
        directory = o.variables[0]
        assert len(directory) == total_features
        actives = directory[directory >= 0]
        assert len(actives) == subset_size
        assert np.count_nonzero(directory >= bins) == 0
        
        if variator_name == 'active_swap':
            orig_directory = orig_offspring_vars[i]
            assert np.all(np.where(directory  < 0)[0] == np.where(orig_directory < 0)[0])
    
    for i, sol in enumerate(parent_sol):
        assert np.all(sol.variables[0] == orig_parent_vars[i])

# @pytest.mark.timeout(2)
def test_set_partition_mutation(set_partition_with_mutation, request):
    
    _, offspring_sol, _= create_one_var_solutions(set_partition_with_mutation)
    total_features, subset_size, bins = request.getfixturevalue("set_partition_dim")
    variator_name = request.getfixturevalue("set_partition_mutator")

    orig_directory = np.copy(offspring_sol.variables[0])
    set_partition_with_mutation.local_variator.mutate(set_partition_with_mutation, offspring_sol, variable_index=0)

    directory = offspring_sol.variables[0]
    assert len(directory) == total_features
    actives = directory[directory >= 0]
    assert len(actives) == subset_size
    assert np.count_nonzero(directory >= bins) == 0

    if variator_name == 'active':
        assert np.all(np.where(directory  < 0)[0] == np.where(orig_directory < 0)[0])

def test_weighted_set_rand(weighted_set_dim):
    
    total_features, min_subset_size, max_subset_size, min_weight, max_weight = weighted_set_dim
    weighted_set =  WeightedSet(total_features, min_subset_size, max_subset_size, min_weight, max_weight)
    min_weight = weighted_set.min_weight 
    max_weight = weighted_set.max_weight
    
    directory = weighted_set.rand()
    # print(f"directory {directory}, min_weight {min_weight}, max_weight {max_weight}")
    
    actives = directory[directory > 0]
    nactives = len(actives)
    ninactives = total_features - nactives
    
    assert len(directory) == total_features
    assert min_subset_size <= nactives <= max_subset_size
    assert np.count_nonzero(directory < min_weight) == ninactives
    assert np.count_nonzero(directory > max_weight) == 0
    assert np.isclose(np.sum(directory), 1.0, rtol = 1e-5, atol = 1e-6)

# @pytest.mark.timeout(5)
def test_weighted_set_crossover(weighted_set_with_crossover, nsolutions_crossover, deepcopy_parents, request):
    nparents, offspring = nsolutions_crossover
    variator_name = request.getfixturevalue("weighted_set_variator")
    parent_sol, offspring_sol, copy_indices = create_one_var_solutions(weighted_set_with_crossover, nparents, offspring, deepcopy_parents)
    
    orig_offspring_vars =  [np.copy(sol.variables[0]) for sol in offspring_sol]
    orig_parent_vars = [np.copy(sol.variables[0]) for sol in parent_sol]

    weighted_set_with_crossover.local_variator.evolve(weighted_set_with_crossover, parent_sol, offspring_sol, variable_index=0, copy_indices=copy_indices)
    for i, o in enumerate(offspring_sol):
        directory = o.variables[0]
        actives = np.where(directory > 0)[0]
        nactives = len(actives)
        ninactives = weighted_set_with_crossover.total_features - nactives
        
        assert len(directory) == weighted_set_with_crossover.total_features
        assert weighted_set_with_crossover.min_features <= nactives <= weighted_set_with_crossover.max_features
        assert np.count_nonzero(directory < weighted_set_with_crossover.min_weight) == ninactives
        assert np.count_nonzero(directory > weighted_set_with_crossover.max_weight) == 0
        assert np.isclose(np.sum(directory), 1.0, rtol = 1e-5, atol = 1e-6)
        
        if variator_name == 'weight':
            orig_directory = orig_offspring_vars[i]
            active_before = np.where(orig_directory > 0)[0]
            assert np.all(actives == active_before)
        
    for i, p in enumerate(parent_sol):
        assert np.all(p.variables[0] == orig_parent_vars[i])
        
    
def test_weighted_set_mutation(weighted_set_with_mutation, request):
    _, offspring_sol, _ = create_one_var_solutions(weighted_set_with_mutation)
    orig_directory = np.copy(offspring_sol)
    weighted_set_with_mutation.local_variator.mutate(weighted_set_with_mutation, offspring_sol, variable_index=0)
    
    directory = offspring_sol.variables[0]
    actives = np.where(directory > 0)[0]
    nactives = len(actives)
    ninactives = weighted_set_with_mutation.total_features - nactives
    
    assert len(directory) == weighted_set_with_mutation.total_features
    assert weighted_set_with_mutation.min_features <= nactives <= weighted_set_with_mutation.max_features
    assert np.count_nonzero(directory < weighted_set_with_mutation.min_weight) == ninactives
    assert np.count_nonzero(directory > weighted_set_with_mutation.max_weight) == 0
    assert np.isclose(np.sum(directory), 1.0, rtol = 1e-5, atol = 1e-6)