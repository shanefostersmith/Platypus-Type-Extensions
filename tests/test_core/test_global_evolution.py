import pytest
from copy import deepcopy
from platypus import RandomGenerator, Integer, Real
from tests.conftest import create_basic_global_evolution
from tests.test_core.conftest import *
from tests.test_core.confglobal import *
from custom_types.core import CustomType, PlatypusType
from custom_types.global_evolutions.global_differential import GlobalDifferential
from custom_types.lists_and_ranges.lists_ranges import RealList, RealListDE, RealListPM

class TestBasicGlobal:
    
    def test_init_and_functions(self, multi_custom_types: Problem, nparents_and_noffspring, global_copy_method):
        basic_global_evolution = create_basic_global_evolution(
            arity = nparents_and_noffspring[0],
            offspring=nparents_and_noffspring[1],
            copy_method= global_copy_method)
        
        
        custom_types = [multi_custom_types.types[i] for i in range(multi_custom_types.nvars)]
        # sprint(f"custom_types: {custom_types}")
        for t in custom_types:
            if isinstance(t, CustomType):
                assert t.can_evolve, f"{type(t)}: variator_type {type(t.local_variator)}"
        
        ngenerics = sum([not isinstance(t, CustomType) for t in custom_types])
        seen = 0
        for custom, i in basic_global_evolution.generate_types_to_evolve(multi_custom_types):
            seen += 1
            assert i < multi_custom_types.nvars
        assert seen ==  multi_custom_types.nvars - ngenerics, "Did not ignore generics"
        
        assert basic_global_evolution._indices_by_type is None
        for type_list, type_indices in basic_global_evolution.generate_type_groups(multi_custom_types):
            assert len(type_list) == len(type_indices)
            prev_type = type_list[0]
            prev_idx = type_indices[0]
            assert prev_type is multi_custom_types.types[prev_idx], "Got the wrong indices"
    
            ntypes = len(type_list)
            if ntypes > 1:
                assert ntypes == len(np.unique(list(type_indices)))
                first_type = type(prev_type)
                for j in range(1, ntypes):
                    next_type = type_list[j]
                    next_idx = type_indices[j]
                    assert type(next_type) == first_type, "Not all the same types were returned"
                    assert next_type is multi_custom_types.types[next_idx]
        assert basic_global_evolution._indices_by_type is not None
        
        rand_generator = RandomGenerator()
        parent_solutions = [rand_generator.generate(multi_custom_types) for _ in range(nparents_and_noffspring[0])]
        for sol in parent_solutions:
            sol.evaluated = True
        
        original_parent_vars = [[deepcopy(sol.variables[i]) for i in range(multi_custom_types.nvars)] for sol in parent_solutions]
        offspring_solutions = basic_global_evolution.evolve(parent_solutions)
        assert len(offspring_solutions) == nparents_and_noffspring[1]
        for i,sol in enumerate(parent_solutions): # ensure no change in parent
            assert all(sol.evaluated for sol in parent_solutions)
            for j in range(multi_custom_types.nvars):
                if isinstance(sol.variables[j], np.ndarray):
                    assert np.all(sol.variables[j] == original_parent_vars[i][j]), f"Parent {i}; the {j}th variable was altered"
                else:
                    assert sol.variables[j] == original_parent_vars[i][j], f"Parent {i}; the {j}th variable was altered"
        
        if global_copy_method == 0:
            if not np.all(offspring_solutions[0].variables[0] == original_parent_vars[0][0]):
                assert not offspring_solutions[0].evaluated 
        
        if not ngenerics:
            evolution_only = set(
                [i for i, t in enumerate(custom_types) 
                if isinstance(t, CustomType) 
                and (not t.do_mutation and t.do_evolution)])
            
            for t in custom_types:
                t.do_evolution = False
                
            for _, i in basic_global_evolution.generate_types_to_evolve(multi_custom_types):
                assert not i in evolution_only, f"fail type {type(multi_custom_types.types[i])}: variator {multi_custom_types.types[i].local_variator!r}"
    
      
def test_global_differential():
    global_de = GlobalDifferential()
    generic_int_1 = Integer(2, 10)
    generic_int_2 = Integer(-5, -1)
    generic_real_1 = Real(0.1, 0.9)
    generic_real_2 = Real(0.5, 0.6)
    real_list1 = RealList([-0.1, 0.1, 0.2], local_variator=RealListDE(0.99))
    real_list2 = RealList([-0.1, 0.1, 0.2], local_mutator=RealListPM())
    mixed_type1 = unconstrainedProblem(
        generic_int_1, generic_int_2,
        generic_real_1, generic_real_2,
        real_list1, real_list2)
    
    parents = create_multi_var_solutions(
        nsolutions = 4,
        problem = mixed_type1
    )
    offspring = global_de.evolve(parents)
    assert isinstance(offspring, list) and len(offspring) == 1
        
        


    