import pytest
from platypus import EvolutionaryStrategy, NSGAII, Integer, Real
from tests.conftest import create_basic_global_evolution
from tests.test_core.conftest import *

@pytest.fixture(
    params = ['one_type', 'mixed', 'compound', 'mutation_only', 'with_generic'],
    ids = lambda v: f"basic_problem_types={v}"
)
def multi_custom_types(request) -> Problem:
    problem_type = request.param
    custom_types = []
    if problem_type == 'one_type':
        custom_types = [
            create_multi_real(MultiRealPM(0.99)), 
            create_multi_real(MultiPCX(0.99)),
            create_multi_real(ArrayCrossover(0.99))
        ]
    elif problem_type == 'mutation_only':
        custom_types = [
            create_multi_real(MultiRealPM(0.99)), 
            RealList([-1, 0, 2, 4], local_mutator = LocalCompoundMutator(local_mutators = [RealListPM(0.99), StepMutation(0.99)])),
            SteppedRange(-1, 2, 0.1, local_mutator = StepMutation(0.99))
        ]
    elif problem_type == 'mixed':
        shared = ArrayCrossover(0.99)
        custom_types  = [ 
            create_multi_real(MultiPCX(0.99)),
            create_multi_real(shared),
            create_multi_int(shared),
            RealList([-1, 0, 2, 4], local_mutator = LocalCompoundMutator(local_mutators = [RealListPM(0.99), StepMutation(0.99)])),
        ]
    elif problem_type == 'compound':
        shared = LocalSequentialOperator(local_variators = [ArrayCrossover(0.99), ArrayCrossover(0.99)])
        shared2 = LocalCompoundMutator(local_mutators = [StepMutation(0.99), StepMutation(0.99)])
        custom_types  = [ 
            create_multi_real(shared),
            create_multi_real(LocalGAOperator(shared, MultiRealPM(0.99))),
            create_multi_int(shared),
            RealList([-1, 0, 2, 4], LocalGAOperator(local_variator = RealListPCX(0.99), local_mutator = shared2)),
            SteppedRange(-1, 2, 0.1, local_mutator = shared2),
        ]
    elif 'with_generic':
        shared = ArrayCrossover(0.99)
        custom_types  = [ 
            create_multi_real(shared),
            create_multi_int(shared),
            Integer(-1, 1),
            Real(-1.0, 1.0)
        ]
    return unconstrainedProblem(*custom_types)

@pytest.fixture(params = [0, 'sample', 'rand'], ids = lambda v: f"copy_method={v}")
def global_copy_method(request):
    return request.param

@pytest.fixture
def basic_global_with_nsgaii(multi_custom_types, nparents_and_noffspring, global_copy_method):
    basic_global_evolution = create_basic_global_evolution(
        arity = nparents_and_noffspring[0],
        offspring=nparents_and_noffspring[1],
        copy_method= global_copy_method)
    
    problem = multi_custom_types
    algorithm = NSGAII(problem, population_size = 8, variator=basic_global_evolution)
    return algorithm

@pytest.fixture
def basic_global_with_evolutionary_strategy(global_copy_method):
    custom_types = [
        create_multi_real(MultiRealPM(0.99)), 
        RealList([-1, 0, 2, 4], local_mutator = LocalCompoundMutator(local_mutators = [RealListPM(0.99), StepMutation(0.99)])),
        SteppedRange(-1, 2, 0.1, local_mutator = StepMutation(0.99))
    ]
    problem = unconstrainedProblem(custom_types)
    basic_global_evolution = create_basic_global_evolution(
        arity = 1,
        offspring= 1,
        copy_method = global_copy_method)
    algorithm = EvolutionaryStrategy(problem = problem, population_size = 10, offpsring_size = 5, variator = basic_global_evolution)
    return algorithm

    