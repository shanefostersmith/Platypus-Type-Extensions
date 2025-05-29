import pytest
from custom_types.lists_and_ranges.lists_ranges import *
from tests.conftest import single_variation_type

@pytest.fixture
def category_type(single_variation_type):
    categories = ['a', 'b', 'c']
    if single_variation_type == 'variate':
        variator = CategoryCrossover(.99)
        return Categories(categories, local_variator=variator)
    elif single_variation_type == 'mutate':
        variator = CategoryMutation(.99)
        return Categories(categories, local_mutator=variator)

@pytest.fixture(params=['de', 'pcx'], ids=lambda v: f"real_list_variation={v}" )
def real_variator(request):
    return request.param

@pytest.fixture(params=['array', 'bits'], ids=lambda v: f"multi_int_variation={v}")
def multi_int_variator(request):
    return request.param

@pytest.fixture(params=['step', 'pm'],ids=lambda v: f"real_list_mutation={v}")
def real_list_mutator(request):
    return request.param

@pytest.fixture
def real_list_with_crossover(real_variator):
    reals = [0.0, -0.8, 1]
    variator = RealListDE(0.99, 0.99) if real_variator == 'de' else RealListPCX(0.99)
    return RealList(reals, local_variator=variator)

@pytest.fixture
def real_list_with_mutation(real_list_mutator):
    reals = [0.0, -0.8, 1]
    variator = StepMutation(.99) if real_list_mutator == 'step' else RealListPM(0.99)
    return RealList(reals, local_mutator=variator)

@pytest.fixture(
    params=['one_step', 'exact_step', 'between_step'],
    ids=lambda v: f"step_bounds={v}"
)
def step_bounds(request):
    """
    Returns
        tuple: lower_bound, upper_bound, step_size
    """
    if request.param == 'one_step':
        return -0.5, 1, 0.5
    elif request.param == 'exact_step':
        return -0.5, 1, 1.5
    else:
        return -0.5, 1, 0.6

@pytest.fixture
def step_value_type(single_variation_type, step_bounds):
    lb, ub, step = step_bounds
    if single_variation_type == 'variate':
        variator = SteppedRangeCrossover(.99)
        return SteppedRange(lb, ub, step, local_variator=variator)
    elif single_variation_type == 'mutate':
        variator = StepMutation(.99)
        return SteppedRange(lb, ub, step, local_mutator=variator)
    
def createMultiRealRanges():
    col1 = np.random.uniform(-1.0, 0.25, size = 4)
    col2 = np.random.uniform(0.5, 1.25, size = 4)
    return np.column_stack((col1, col2))

def createMultiIntRanges():
    col1 = np.random.randint(-4, 1, size=4)
    col2 = np.random.randint(1, 5, size=4)
    arr = np.column_stack((col1, col2))
    arr[0] = [0, 0]
    return arr.astype(dtype = np.int16)
    
@pytest.fixture
def multi_real_with_crossover(real_variator):
    variator = MultiDifferentialEvolution(0.99, 0.99) if real_variator == 'de' else MultiPCX(0.99)
    return MultiRealRange(createMultiRealRanges(), local_variator=variator)

@pytest.fixture
def multi_real_with_mutation():
    return MultiRealRange(createMultiRealRanges(), local_mutator=MultiRealPM(0.99))

@pytest.fixture
def multi_int_with_crossover(multi_int_variator):
    variator = MultiIntegerCrossover(0.99) if multi_int_variator == 'bits' else ArrayCrossover(0.99)
    return MultiIntegerRange(createMultiIntRanges(), local_variator = variator)

@pytest.fixture
def multi_int_with_mutation():
    return MultiIntegerRange(createMultiIntRanges(), local_mutator = MultiIntegerMutation(0.99))
