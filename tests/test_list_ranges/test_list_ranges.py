import pytest
import numpy as np
from tests.test_list_ranges.test_list_ranges import *
from custom_types.lists_and_ranges.lists_ranges import MultiDifferentialEvolution, RealListDE
from tests.conftest import create_one_var_solutions
from platypus import Problem, Solution

def test_categories(category_type):
    parent_sol, offspring_sol, _ = create_one_var_solutions(category_type)
    
    if parent_sol is None:
        category_type.local_variator.mutate(category_type, offspring_sol, variable_index=0) 
    else:
        orig_parent_categories = [sol.variables[0] for sol in parent_sol]
        category_type.local_variator.evolve(category_type, parent_sol, offspring_sol, variable_index=0, copy_indices = [None])
        for i, sol in enumerate(parent_sol):
            assert sol.variables[0] == orig_parent_categories[i]

def test_step_range(step_value_type):
    parent_sol, offspring_sol, _ = create_one_var_solutions(step_value_type)
    
    if parent_sol is None:
        step_value_type.local_variator.mutate(step_value_type, offspring_sol, variable_index=0) 
    else:
        orig_parent_categories = [sol.variables[0] for sol in parent_sol]
        step_value_type.local_variator.evolve(step_value_type, parent_sol, offspring_sol, variable_index=0, copy_indices = [None])
        for i, sol in enumerate(parent_sol):
            assert sol.variables[0] == orig_parent_categories[i]

def test_multi_real_crossover(multi_real_with_crossover):
    nparents = 2
    noffspring = 2
    if isinstance(multi_real_with_crossover.local_variator, MultiDifferentialEvolution):
        nparents = 4
        noffspring = 1
        
    parent_sol, offspring_sol, copy_indices = create_one_var_solutions(multi_real_with_crossover, nparents, noffspring, deepcopy=True)
    orig_parent_categories = [sol.variables[0] for sol in parent_sol]
    multi_real_with_crossover.local_variator.evolve(multi_real_with_crossover, parent_sol, offspring_sol, variable_index=0, copy_indices = copy_indices)
    for i, sol in enumerate(parent_sol):
        assert np.all(sol.variables[0] == orig_parent_categories[i])

def test_multi_real_mutation(multi_real_with_mutation):
    _, offspring_sol, _ = create_one_var_solutions(multi_real_with_mutation)
    multi_real_with_mutation.local_variator.mutate(multi_real_with_mutation, offspring_sol, variable_index=0)

def test_real_list_crossover(real_list_with_crossover):
    nparents = 2
    noffspring = 2
    if isinstance(real_list_with_crossover.local_variator, RealListDE):
        nparents = 4
        noffspring = 1
        
    parent_sol, offspring_sol, copy_indices = create_one_var_solutions(real_list_with_crossover, nparents, noffspring, True)
    orig_parent_vars = [sol.variables[0] for sol in parent_sol]
    real_list_with_crossover.local_variator.evolve(real_list_with_crossover, parent_sol, offspring_sol, variable_index=0, copy_indices = copy_indices)
    for i, sol in enumerate(parent_sol):
        assert sol.variables[0] == orig_parent_vars[i]

def test_real_list_mutation(real_list_with_mutation):
    _, offspring_sol, _ = create_one_var_solutions(real_list_with_mutation)
    real_list_with_mutation.local_variator.mutate(real_list_with_mutation, offspring_sol, variable_index=0)

def test_multi_int_crossover(multi_int_with_crossover):
    parent_sol, offspring_sol, _ = create_one_var_solutions(multi_int_with_crossover, nparents = 3, noffspring = 2)
    orig_parent_vars = [sol.variables[0] for sol in parent_sol]
    multi_int_with_crossover.local_variator.evolve(multi_int_with_crossover, parent_sol, offspring_sol, variable_index=0, copy_indices = [None, 1])
    for i, sol in enumerate(parent_sol):
        assert np.all(sol.variables[0] == orig_parent_vars[i])
    
    
    
    
    
