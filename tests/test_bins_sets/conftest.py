import pytest
from platypus_extensions.bins_and_sets.bins_and_sets import *

@pytest.fixture( params=[ (2,1), (2,2), (2,4) ], ids=lambda v: f"nsolutions={v}" )
def nsolutions_crossover(request):
    return request.param

@pytest.fixture( params = ['rand', 'equal'], ids=lambda v: f"set_partition_init={v}" )
def set_partition_start(request):
    return request.param

@pytest.fixture(
    params = [
        (4, 3, 2),
        (4, 2, 3),
        (4, 4, 4),
        (2, 2, 2),
        (6, 2, 2)
    ],
    ids=lambda v: f"set_partition_dim={v}"
)
def set_partition_dim(request):
    """ 
    Returns
        tuple: (total_features, subset_size, bins)"""
    return request.param

@pytest.fixture(params=['active_swap', 'subset_swap'], ids=lambda v: f"partition_variation={v}")
def set_partition_variator(request):
    return request.param

@pytest.fixture(params=['active', 'subset', 'combined'], ids=lambda v: f"partition_mutation={v}")
def set_partition_mutator(request):
    return request.param

@pytest.fixture
def set_partition_with_crossover(set_partition_variator, set_partition_dim):
    total_features, subset_size, bins = set_partition_dim
    variator = ActiveBinSwap(0.99) if set_partition_variator == 'active_swap' else FixedSubsetSwap(0.99)
    return SetPartition(total_features, subset_size, bins, local_variator=variator)

@pytest.fixture
def set_partition_with_mutation(set_partition_mutator, set_partition_dim):
    total_features, subset_size, bins = set_partition_dim
    bin_mutation_prob = 0 if set_partition_mutator == 'subset' else 0.999
    bin_swap_rate = 0 if set_partition_mutator == 'active' else 0.999
    return SetPartition(total_features, subset_size, bins, local_mutator = BinMutation(bin_mutation_prob, bin_swap_rate))


@pytest.fixture(params=['combined', 'subset', 'weight'],
ids=lambda v: f"weighted_set_variation={v}")
def weighted_set_variator(request):
    return request.param

@pytest.fixture(
    params = [
        (5, 3, 5, 0.1, 0.9),
        (5, 3, 4, 0.25, 0.5),
        (5, 2, 4, 0.25, 0.5),
        (5, 3, 5, 0.19, 0.35),
        (5, 4, 4, 0.2, 0.3),
        (4, 4, 4, 0.26, 0.5),
        (2, 2, 2, 0.1, 0.99),
        # (500, 2, 500, 0.000005, 0.9999)
    ],
    ids=lambda v: f"weighted_set_dim={v}"
)
def weighted_set_dim(request):
    """
    Returns
        tuple: (total_features, min_subset_size, max_subset_size, min_weight, max_weight)
        """
    return request.param

@pytest.fixture
def weighted_set_with_crossover(weighted_set_dim, weighted_set_variator):
    total_features, min_subset_size, max_subset_size, min_weight, max_weight = weighted_set_dim
    weight_crossover_rate = 0 if weighted_set_variator == 'subset' else 0.999
    subset_evolve_prob = 0 if weighted_set_variator == 'weight' else 0.999
    return WeightedSet(total_features, min_subset_size, max_subset_size, min_weight, max_weight, local_variator=WeightedSetCrossover(subset_evolve_prob, weight_crossover_rate))

@pytest.fixture
def weighted_set_with_mutation(weighted_set_dim):
    total_features, min_subset_size, max_subset_size, min_weight, max_weight = weighted_set_dim
    return WeightedSet(total_features, min_subset_size, max_subset_size, min_weight, max_weight, local_mutator=WeightedSetMutation(0.999, 0.999))
