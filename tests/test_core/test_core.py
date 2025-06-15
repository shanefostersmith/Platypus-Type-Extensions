import pytest
from custom_types.core import *
from tests.test_core.conftest import *
from custom_types.global_evolutions.general_global_evolution import GeneralGlobalEvolution

@pytest.mark.parametrize(
    'variators', [
     ArrayCrossover(), 
     StepMutation(),
     [ArrayCrossover(), ArrayCrossover()],
     [MultiDifferentialEvolution(), ArrayCrossover()],
     [ArrayCrossover(), MultiIntegerMutation()], 
     [StepMutation(), StepMutation(), StepMutation()],
     [StepMutation(), StepMutation(), RealListPM()],
     [ArrayCrossover(), ArrayCrossover(), MultiIntegerMutation()],
    ]
)
def test_type_tuple(variators):
    to_test = None
    if isinstance(variators, LocalVariator):
        to_test = variators
    elif len(variators) == 2 and isinstance(variators[1], LocalMutator):
        to_test = LocalGAOperator(variators[0], variators[1])
    elif len(variators) == 2:
        to_test = LocalSequentialOperator(local_variators= variators)
    else:
        seq = LocalSequentialOperator(local_variators= [variators[0], variators[1]])
        assert isinstance(seq._supported_types, TypeTuple), f"{type(seq._supported_types)}"
        for t in seq._supported_types:
            assert isinstance(t, type), f"Main Type {type(seq)}: Inner type {type(t)} for Variators: \n {variators}"
            assert t != ABCMeta, f"{type(t)} for variators {variators}"
            assert t != TypeTuple, f"{type(t)} for variators {variators}"
            
        to_test = LocalGAOperator(seq, variators[-1])
    
    assert isinstance(to_test._supported_types, TypeTuple), f"Tuple Type: {type(to_test._supported_types)} for variators {variators}"
    for t in to_test._supported_types:
        assert isinstance(t, type), f"Main Type {type(to_test)}: Inner type {type(t)} for Variators: \n {variators}"
        assert t != ABCMeta, f"Main Type {type(to_test)}: Inner type {t} for Variators: \n {variators}"
        assert t != TypeTuple, f"Main Type {type(to_test)}: Inner type {t} for Variators: \n {variators}"
    
    if isinstance(to_test, ArrayCrossover):
        ArrayCrossover.register_type(RealList)
        assert RealList in ArrayCrossover._supported_types
        assert RealList in to_test._supported_types
        assert MultiIntegerRange in ArrayCrossover._supported_types and MultiRealRange in ArrayCrossover._supported_types

        with pytest.raises(TypeError):
            ArrayCrossover.register_type((SetPartition, SteppedRange))

def test_sequential_operator_single(single_selection_partition_solutions, request):
    """Tests LocalSequentialOperator with SetPartition. Uses 3-4 variators, 1 is a LocalMutator """
    parent_solutions, offspring_solutions, copy_indices = single_selection_partition_solutions
    nparents, noffspring = request.getfixturevalue("nparents_and_noffspring")
    print(f"nparents, noffspring = ({nparents, noffspring})")
    assert len(copy_indices) == len(offspring_solutions)
    
    problem = parent_solutions[0].problem
    custom_type = problem.types[0]
    sequential_variator: LocalSequentialOperator = custom_type.local_variator
    
    assert custom_type.do_evolution and custom_type.do_mutation 
    assert SetPartition in custom_type.local_variator._supported_types
    assert sequential_variator._contains_crossover and sequential_variator._contains_mutation
    assert sequential_variator._supported_arity[0] == 2 
    assert sequential_variator._supported_noffspring[0] == 1
    
    if sequential_variator.offspring_selection == 'swap':
        assert sequential_variator.contains_swap == len(sequential_variator.variators) - 1
    
    orig_parent_vars = [np.copy(sol.variables[0]) for sol in parent_solutions]
    
    custom_type.local_variator.evolve(custom_type, parent_solutions, offspring_solutions, variable_index = 0, copy_indices = copy_indices)
    assert len(parent_solutions) == nparents
    assert len(offspring_solutions) == noffspring
    
    for i, p in enumerate(parent_solutions):
        assert p.evaluated
        assert np.all(p.variables[0] == orig_parent_vars[i])

def test_sequential_operator_multi(multi_selection_partition_solutions, request):
    parent_solutions, offspring_solutions, copy_indices = multi_selection_partition_solutions
    nparents, noffspring = request.getfixturevalue("nparents_and_noffspring")
    print(f"nparents, noffspring = ({nparents, noffspring})")
    
    problem = parent_solutions[0].problem
    custom_type = problem.types[0]
    sequential_variator: LocalSequentialOperator = custom_type.local_variator
    assert custom_type.do_evolution and custom_type.do_mutation 
    assert SetPartition in custom_type.local_variator._supported_types
    assert sequential_variator._contains_crossover and sequential_variator._contains_mutation
    
    orig_parent_vars = [np.copy(sol.variables[0]) for sol in parent_solutions]
    
    custom_type.local_variator.evolve(custom_type, parent_solutions, offspring_solutions, variable_index = 0, copy_indices = copy_indices)
    assert len(parent_solutions) == nparents
    assert len(offspring_solutions) == noffspring
    
    for i, p in enumerate(parent_solutions):
        assert p.evaluated
        assert np.all(p.variables[0] == orig_parent_vars[i])

def test_sequential_operator_mixed(single_selection_real_solutions, request):
    
    parent_solutions, offspring_solutions, copy_indices = single_selection_real_solutions
    nparents, noffspring = request.getfixturevalue("nparents_and_noffspring")
    print(f"nparents, noffspring = ({nparents, noffspring})")
    
    problem = parent_solutions[0].problem
    custom_type = problem.types[0]
    assert custom_type.can_evolve
    sequential_variator: LocalSequentialOperator = custom_type.local_variator
    
    with_differential = False
    if isinstance(sequential_variator.variators[0], MultiRealPM):
        assert not sequential_variator._contains_crossover and sequential_variator._contains_mutation
    elif isinstance(sequential_variator.variators[0], ArrayCrossover):
        assert sequential_variator._contains_crossover and not sequential_variator._contains_mutation
        assert custom_type.do_evolution and not custom_type.do_mutation
        assert MultiRealRange in sequential_variator._supported_types and MultiIntegerRange in sequential_variator._supported_types
    else:
        with_differential = True
        assert sequential_variator._supported_arity[0] == 4
        assert sequential_variator._supported_noffspring[0] == 1
        assert MultiRealRange in sequential_variator._supported_types and not MultiIntegerRange in sequential_variator._supported_types, f"types: {sequential_variator._supported_types}"
    
    orig_parent_vars = [np.copy(sol.variables[0]) for sol in parent_solutions]
    
    if with_differential and nparents < 4:
        with pytest.raises(ValueError):
            custom_type.local_variator.evolve(custom_type, parent_solutions, offspring_solutions, variable_index = 0, copy_indices = copy_indices)
        assert all(sol.evaluated for sol in offspring_solutions) # Return immediately
    else:
        custom_type.local_variator.evolve(custom_type, parent_solutions, offspring_solutions, variable_index = 0, copy_indices = copy_indices)
    
    for i, p in enumerate(parent_solutions):
        assert p.evaluated
        assert np.all(p.variables[0] == orig_parent_vars[i])
        
def test_gao_operator_mixed(gao_operator_solutions, request):
    
    parent_solutions, offspring_solutions, copy_indices = gao_operator_solutions
    nparents, noffspring = request.getfixturevalue("nparents_and_noffspring")
    print(f"nparents, noffspring = ({nparents, noffspring})")
    
    problem = parent_solutions[0].problem
    custom_type = problem.types[0]
    assert custom_type.can_evolve
    gao: LocalGAOperator = custom_type.local_variator
    
    orig_parent_vars = [np.copy(sol.variables[0]) for sol in parent_solutions]
    if gao._supported_arity[0] == 4 and nparents < 4:
        with pytest.raises(ValueError):
            custom_type.local_variator.evolve(custom_type, parent_solutions, offspring_solutions, variable_index = 0, copy_indices = copy_indices)
        assert all(sol.evaluated for sol in offspring_solutions) # Return immediately
    else:
        for i, p in enumerate(parent_solutions):
            assert p.evaluated
            assert np.all(p.variables[0] == orig_parent_vars[i])

def test_local_compound_mutator(compound_mutator_solutions: list[Solution]):
    problem = compound_mutator_solutions[0].problem
    custom_type = problem.types[0]
    mutator: LocalCompoundMutator = custom_type.local_variator

    mutator.evolve(custom_type, [], compound_mutator_solutions, variable_index=0, copy_indices=[])
    mutator.mutate(custom_type, compound_mutator_solutions[0], variable_index=0)
 
def test_variator_resets():
    real_list = RealList([-1, 0, 4])
    assert not (real_list.do_evolution or real_list.do_mutation)
    assert not real_list.can_evolve
    
    mutator = LocalCompoundMutator([StepMutation(), StepMutation()])
    real_list.local_variator = mutator
    assert not real_list.do_evolution and real_list.do_mutation
    assert real_list.can_evolve
    
    real_list.local_variator = None
    assert not (real_list.do_evolution or real_list.do_mutation)
    assert not real_list.can_evolve
    
    variator = RealListPCX()
    real_list.local_variator = LocalGAOperator(variator, mutator) 
    assert real_list.do_evolution and real_list.do_mutation
    assert real_list.can_evolve
    
    real_list.local_variator = variator
    assert real_list.do_evolution and not real_list.do_mutation
    assert real_list.can_evolve
    
    
