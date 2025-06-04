
import pytest
import copy
from itertools import permutations
from custom_types.core import *
from tests.test_core.conftest import *
from custom_types.bins_and_sets.bins_and_sets import SetPartition, ActiveBinSwap, FixedSubsetSwap, BinMutation
from custom_types.lists_and_ranges.lists_ranges import (
    MultiRealRange, MultiIntegerRange, RealList, SteppedRange,
    ArrayCrossover, MultiIntegerMutation,
    MultiPCX, MultiDifferentialEvolution, MultiRealPM,
    RealListPCX, StepMutation, RealListPM)
from tests.conftest import (
    create_basic_global_evolution, 
    create_multi_var_solutions, 
    create_one_var_solutions,
    unconstrainedProblem)

SELECTION_STRATEGIES = ['previous', 'swap', 'rand']
THREE_STRATEGIES = [p for p in permutations(SELECTION_STRATEGIES) if p[0] != 'swap']

@pytest.fixture(params = SELECTION_STRATEGIES, ids = lambda v: f"offspring_selection={v}") #SELECTION_STRATEGIES
def single_offspring_selection(request):
    return request.param

@pytest.fixture(params = [(2, 1), (2,4), (4,4), (5,2)], ids = lambda v: f"nparents={v[0]}, noffspring={v[1]}")
def nparents_and_noffspring(request):
    return request.param

@pytest.fixture(params = [True, False], ids = lambda v: f"mutation_first={v}")
def mutation_first(request):
    return request.param

@pytest.fixture(params=[None, 'order', 'mixed'], ids = lambda v: f"deepcopy_type={v}")
def deepcopy_type(request):
    return request.param

@pytest.fixture( params = THREE_STRATEGIES, ids = lambda v: f"selection_order={v}" )
def multi_selection_set_partition(request, mutation_first):
    selection_strategies = list(request.param)
    variators = []
    if mutation_first:
        variators.append(BinMutation(0.99, 0.99))
    else:
        variators.append(ActiveBinSwap(0.999))

    variators.append(FixedSubsetSwap(0.999))
    if not mutation_first:
        variators.append(BinMutation(0.99, 0.99))
        if selection_strategies[-1] == 'swap':
            selection_strategies[-1] = 'all'
    else:
        variators.append(ActiveBinSwap(0.999))
    
    return LocalSequentialOperator(local_variators=variators, offspring_selection=selection_strategies)

@pytest.fixture(params = [True, False], ids = lambda v: f"randomize_first_op={v}")
def randomize_first_operator(request):
    return request.param

@pytest.fixture
def single_selection_set_partition(single_offspring_selection, randomize_first_operator):
    variators = []
    mutation_strategy = single_offspring_selection if single_offspring_selection != 'swap' else 'all'
    variators.append(ActiveBinSwap(0.999))
    variators.append(FixedSubsetSwap(0.999))
    variators.append(BinMutation(0.99, 0.99))
    
    return LocalSequentialOperator(
        local_variators=variators, 
        offspring_selection=single_offspring_selection,
        mutation_selection = mutation_strategy,
        randomize_start = randomize_first_operator
    )

@pytest.fixture(params = ["array_crossover_only", "mutation_only", "mixed", "all"], ids = lambda v: "variator_combo = {v}")
def single_selection_multi_real(request, single_offspring_selection, randomize_first_operator):
    variation_type = request.param
    variators = []
    mutation_selection = 'rand'
    if variation_type == "array_crossover_only":
        variators = [ArrayCrossover(0.999), ArrayCrossover(0.999)]
    elif variation_type == 'all':
        variators = [MultiPCX(0.999), ArrayCrossover(0.999), MultiDifferentialEvolution(0.999), MultiRealPM(0.999)]
    elif variation_type == 'mixed':
        variators = [MultiDifferentialEvolution(0.999), ArrayCrossover(0.999)]
    else:
        variators = [MultiRealPM(0.999), LocalCompoundMutator(local_mutators= [MultiRealPM(0.999), MultiRealPM(0.999)])]
        mutation_selection = single_offspring_selection if single_offspring_selection != 'swap' else 'all'
    
    sequential_variator = LocalSequentialOperator(
        local_variators=variators, 
        offspring_selection=single_offspring_selection,
        mutation_selection=mutation_selection,
        randomize_start = randomize_first_operator
    )
    if variation_type == 'mutation_only':
        assert not sequential_variator._contains_crossover 
    return sequential_variator
    
def _create_mixed_copy(custom_type, num_parents, num_offspring):
    """Assumes 1 variable"""
    copy_choices = [0, None, 1] if num_parents == 1 or num_parents % 2 == 0 else [None, 0, 1]
    problem = unconstrainedProblem(custom_type)
    
    added_parents = 0
    copy_choices_idx = 0
    parent_solutions = []
    offspring_solutions = []
    copy_indices = []
    for _ in range(num_offspring):
        parent_idx = copy_choices[copy_choices_idx]
        if parent_idx is not None:
            if parent_idx == added_parents:
                parent = Solution(problem)
                parent.variables[0] = custom_type.rand()
                parent.evaluated = True
                parent_solutions.append(parent)
                offspring_solutions.append(copy.deepcopy(parent))
                added_parents += 1
            elif parent_idx < added_parents:
                offspring_solutions.append(copy.deepcopy(parent_solutions[parent_idx]))
            else:
                raise ValueError("error, added_parents should not be > parent_index ")
        else:
            offspring = Solution(problem)
            offspring.variables[0] = custom_type.rand()
            offspring.evaluated = True
            offspring_solutions.append(offspring)
            
        copy_indices.append(parent_idx)
        copy_choices_idx = (1 + copy_choices_idx) % 3
    
    if num_parents > added_parents:
        for _ in range(num_parents - added_parents):
            parent = Solution(problem)
            parent.variables[0] = custom_type.rand()
            parent.evaluated = True
            parent_solutions.append(parent)
    
    return parent_solutions, offspring_solutions, copy_indices 

@pytest.fixture
def multi_selection_partition_solutions(multi_selection_set_partition: LocalSequentialOperator, deepcopy_type, nparents_and_noffspring):
    """Returns: parent_solutions, offspring_solution, copy_indices"""
    set_partition = SetPartition(4, 3, 3, local_variator=multi_selection_set_partition)
    assert set_partition.do_evolution and set_partition.do_mutation
    
    num_parents = nparents_and_noffspring[0]
    num_offspring = nparents_and_noffspring[1]
    if deepcopy_type is None:
        all_solutions = create_multi_var_solutions(num_parents + num_offspring, set_partition)
        copy_indices = [None for _ in range(num_offspring)]
        return all_solutions[:num_parents], all_solutions[num_parents:], copy_indices
    if deepcopy_type == 'order':
        return create_one_var_solutions(set_partition, num_parents, num_offspring, deepcopy=True)
    
    return _create_mixed_copy(set_partition, num_parents, num_offspring)

@pytest.fixture
def single_selection_real_solutions(single_selection_multi_real: LocalSequentialOperator, deepcopy_type, nparents_and_noffspring):
    """Returns: parent_solutions, offspring_solution, copy_indices"""
    input_ranges = np.vstack([np.array([-1, 1], np.float32), np.array([2, 5], np.float32), np.array([-4, 2], np.float32)])
    real_type = MultiRealRange(input_ranges, local_variator=single_selection_multi_real)
    
    num_parents = nparents_and_noffspring[0]
    num_offspring = nparents_and_noffspring[1]
    if deepcopy_type is None:
        all_solutions = create_multi_var_solutions(num_parents + num_offspring, real_type)
        copy_indices = [None for _ in range(num_offspring)]
        return all_solutions[:num_parents], all_solutions[num_parents:], copy_indices
    if deepcopy_type == 'order':
        return create_one_var_solutions(real_type, num_parents, num_offspring, deepcopy=True)
    
    return _create_mixed_copy(real_type, num_parents, num_offspring)
    

@pytest.fixture
def single_selection_partition_solutions(single_selection_set_partition: LocalSequentialOperator, deepcopy_type, nparents_and_noffspring):
    set_partition = SetPartition(4, 3, 3, local_variator=single_selection_set_partition)
    assert set_partition.do_evolution and set_partition.do_mutation
    
    num_parents = nparents_and_noffspring[0]
    num_offspring = nparents_and_noffspring[1]
    
    if deepcopy_type is None:
        all_solutions = create_multi_var_solutions(num_parents + num_offspring, set_partition)
        copy_indices = [None for _ in range(num_offspring)]
        return all_solutions[:num_parents], all_solutions[num_parents:], copy_indices
    
    if deepcopy_type == 'order':
        return create_one_var_solutions(set_partition, num_parents, num_offspring, deepcopy=True)
    
    return _create_mixed_copy(set_partition, num_parents, num_offspring)


@pytest.fixture(params = ['fail', 'compound', 'compound2'], ids = lambda v: f"gao_type={v}")
def gao_operators(request):
    gao_type = request.param
    custom_type = None
    
    if gao_type == 'compound':
        variator = LocalSequentialOperator(
            local_variators= [RealListPCX(0.99), RealListPM(0.99)], 
            offspring_selection= 'rand',
            mutation_selection= 'rand',
        )
        custom_type = RealList([-2, 0, 1, 4], local_variator=LocalGAOperator(variator, StepMutation(0.99)))
        supported = custom_type.local_variator._supported_types
        assert RealList in supported and not SteppedRange in supported 
        
    elif gao_type == 'compound_2':
        inner_variate = LocalSequentialOperator(
            local_variators= [StepMutation(0.99), StepMutation(0.99)], 
            offspring_selection= 'rand',
            mutation_selection= 'all',
        )
        assert RealList in inner_variate._supported_types and SteppedRange in inner_variate._supported_types
        assert all(isinstance(variator_type, type) for variator_type in inner_variate._supported_types)
        
        variator = LocalGAOperator(local_variator=inner_variate, local_mutator=StepMutation(0.99))
        assert RealList in variator._supported_types and SteppedRange in variator._supported_types
        assert all(isinstance(variator_type, type) for variator_type in variator._supported_types )
        
        custom_type = RealList([-2, 0, 1, 4], local_variator=LocalGAOperator(variator, RealListPM(0.99)))
        assert custom_type.local_variator._supported_arity[0] == 1 and custom_type.local_variator._supported_arity[1] == None
        assert RealList in custom_type.local_variator._supported_types and not SteppedRange in custom_type.local_variator._supported_types
        
    else:
        input_ranges = np.vstack([np.array([-1, 1], np.float32), np.array([2, 5], np.float32), np.array([-4, 2], np.float32)])
        with pytest.raises(Exception):
            LocalGAOperator(ArrayCrossover(), BinMutation())
        with pytest.raises(Exception):
            LocalGAOperator(ArrayCrossover(), ArrayCrossover())
        with pytest.raises(Exception):
            MultiRealRange(input_ranges, local_variator=LocalGAOperator(ArrayCrossover(), MultiIntegerMutation()))
            
        input_ranges = np.vstack([np.array([-1, 1], np.float32), np.array([2, 5], np.float32), np.array([-4, 2], np.float32)])
        variator = LocalSequentialOperator(
            local_variators= [MultiDifferentialEvolution(0.99), ArrayCrossover(0.99)], 
            offspring_selection= 'rand',
            mutation_selection= 'rand',
        )
        test_evolve = MultiRealRange(input_ranges, local_variator=variator)
        assert test_evolve.can_evolve
        test_evolve.do_evolution = False
        assert not test_evolve.can_evolve
        test_evolve.do_evolution = True
        assert test_evolve.can_evolve
        
        custom_type = MultiRealRange(input_ranges, local_variator=variator, local_mutator= MultiRealPM(0.99))
        assert isinstance(custom_type.local_variator, LocalGAOperator)
        assert custom_type.local_variator._supported_arity[0] == 4 and custom_type.local_variator._supported_arity[1] == None
    
    assert custom_type.do_evolution
    assert custom_type.do_mutation
    return custom_type

@pytest.fixture
def gao_operator_solutions(gao_operators, nparents_and_noffspring):
    num_parents = nparents_and_noffspring[0]
    num_offspring = nparents_and_noffspring[1]
    return _create_mixed_copy(gao_operators, num_parents, num_offspring)

@pytest.fixture(params = ['fail', 'mixed', 'compound'], ids = lambda v: f"compound_mutator_type={v}")  
def compound_mutator_solutions(request, randomize_first_operator, noffspring):
    test_type = request.param
    mutator = None
    if test_type == 'mixed':
        test_mutator = LocalCompoundMutator(local_mutators=[StepMutation(0.99), StepMutation(0.99)])
        assert RealList in test_mutator._supported_types and SteppedRange in test_mutator._supported_types
        mutator = LocalCompoundMutator(local_mutators=[StepMutation(0.99), RealListPM(0.99), StepMutation(0.99)], randomize_start=randomize_first_operator)
        assert RealList in mutator._supported_types and not SteppedRange in mutator._supported_types

    elif test_type == 'compound':
        inner_mutator1 = LocalCompoundMutator(local_mutators=[StepMutation(0.99), StepMutation(0.99)])
        assert RealList in inner_mutator1._supported_types and SteppedRange in inner_mutator1._supported_types
        
        inner_mutator2 = LocalCompoundMutator(local_mutators=[RealListPM(0.99), StepMutation(0.99)])
        mutator = LocalCompoundMutator(local_mutators=[inner_mutator1, inner_mutator2, RealListPM(0.99)], randomize_start=randomize_first_operator)
        
    else:
        with pytest.raises(Exception):
            LocalCompoundMutator(local_mutators=[StepMutation(), BinMutation()])
        with pytest.raises(Exception):
            LocalCompoundMutator(local_mutators=[RealListPM(), RealListPCX()])
            
        mutator = LocalCompoundMutator(local_mutators= [RealListPM(), RealListPM()], randomize_start=randomize_first_operator)
        with pytest.raises(Exception):
            RealList([-1, 0, 2, 4], local_variator=mutator)
        
    custom_type = RealList([-1, 0, 2, 4], local_mutator=mutator)
    assert mutator._supported_arity[0] == 1
    assert custom_type.can_evolve
    assert not custom_type.do_evolution
    
    solutions = []
    problem = unconstrainedProblem(custom_type)
    for _ in range(noffspring):
        sol = Solution(problem)
        sol.variables[0] = custom_type.rand()
        sol.evaluated = True
        solutions.append(sol)
        
    return solutions