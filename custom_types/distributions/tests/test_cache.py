
import pytest
from time import perf_counter
from custom_types.global_evolutions.general_global_evolution import GeneralGlobalEvolution
from platypus import EvolutionaryStrategy, NSGAII
from pytest_mock import MockerFixture
from tests.conftest import unconstrainedProblem, Solution, CustomType
from .conftest import HALF_LIFE_PARAMS, half_life_bijection
from ..monotonic_distributions import MonotonicDistributions, DistributionShift, DistributionBoundsPCX
from ..point_bounds import PointBounds
from ..real_bijection import RealBijection

NVARS = 3
PROBABILITIES = [0.1, 0.25, 0.5]
USE_CACHE = True
ITERS = 1000
MAX_POINTS = 250

def _create_global_evolution_and_alg(problem, population_size = 2, offspring_size = 2):
    # global_ev = GeneralGlobalEvolution(nparents = 1, noffspring = 1, copy_method=0)
    # alg = EvolutionaryStrategy(problem, population_size, offspring_size, variator=global_ev)
    global_ev = GeneralGlobalEvolution(nparents = 2, noffspring = 2, copy_method='rand')
    alg = NSGAII(problem, population_size, variator=global_ev)
    return alg

def _create_one_bijection():
    bijection_params = HALF_LIFE_PARAMS[0][0]
    forward_func, inverse_func, x_max, _, abs_max_points = half_life_bijection(**bijection_params)
    max_points = min(abs_max_points, MAX_POINTS)
    bounds = PointBounds(
        lower_bound=0.0, 
        upper_bound=x_max, 
        minimum_points = int(0.8*MAX_POINTS),
        maximum_points = max_points,
        precision = 'double')
    bij = RealBijection(forward_func, bounds.create_bounds_state(), inverse_func)
    del bounds
    return bij

def _create_mono_distribtions(cache_type, mutation_probability = 0.99, cache_size = 20):
    bij = _create_one_bijection()
    
    distribs = []
    if cache_type == 'cache':
        for i in range(len(PROBABILITIES)):
            distribution = MonotonicDistributions(
                mappings=[bij], 
                local_variator= DistributionBoundsPCX(PROBABILITIES[i]),
                local_mutator= DistributionShift(PROBABILITIES[i]), #
                use_cache = None if cache_size == 0 else True,
                max_cache_size=cache_size
            )
            distribs.append(distribution)
    
    else:
        print(f"mutation_prob: {mutation_probability}")
        mutator = DistributionShift(mutation_probability) 
        for _ in range(NVARS):
            
            distribution = MonotonicDistributions(
                mappings=[bij], 
                local_mutator=mutator,
                use_cache = None if not USE_CACHE else cache_type,
            )
            distribs.append(distribution)

    return unconstrainedProblem(*distribs)
    
@pytest.fixture(scope = 'function')
def single_memo_distribution(request):
    """Create a problem and algorithm with encoding -> decoding cache
    """
    problem = _create_mono_distribtions('single', 0.999)
    alg = _create_global_evolution_and_alg(problem, 1, 1)
    return problem, alg

@pytest.fixture(params = [(3,2), (20,2), (50, 10)], ids =lambda v: f"population_size={v[0]}, offspring = {v[1]}")
def population_dim(request):
    return request.param

@pytest.fixture(params = [2, 10, 20, 75], ids = lambda v: f"cache_size={v}",)
def max_cache_size(request):
    return request.param

@pytest.fixture(scope = 'function')
def cache_memo_distribution(population_dim, max_cache_size):
    """Create 2 Solution object with 1 CustomType in the Problem w/ a single cache. 
    """
    problem = _create_mono_distribtions('cache', cache_size=max_cache_size)
    alg = _create_global_evolution_and_alg(problem, population_dim[0], population_dim[1])
    return  problem, alg
    

class TestMonoDistributionCache:
    
    def test_memo_decode_to_encode(self, single_memo_distribution: list[Solution], mocker: MockerFixture):
        problem, alg = single_memo_distribution
        problem_spy = mocker.spy(problem, "evaluate")
        alg_spy = mocker.spy(alg, 'iterate')
        
        custom_type: CustomType = problem.types[0]
        # custom_type2: CustomType = problem.types[1]
        # custom_type3: CustomType = problem.types[2]
        
        new_encode_spy1 = mocker.spy(custom_type, "encode")
        # new_encode_spy2 = mocker.spy(custom_type2, "encode")
        # new_encode_spy3 = mocker.spy(custom_type3, "encode")
        orig_encode_spy1 = mocker.spy(custom_type, "_mem_encode")
        # orig_encode_spy2 = mocker.spy(custom_type2, "_mem_encode")
        # orig_encode_spy3 = mocker.spy(custom_type3, "_mem_encode")

        
        print(f"nvars {problem.nvars}")
        start_time = perf_counter()
        alg.run(condition=5)
        end_time = perf_counter()
       
        print(f"nevalutions: {problem_spy.call_count}")
        print(f"niteration: {alg_spy.call_count}")
        print(f"orig encode1 calls: {orig_encode_spy1.call_count} vs new encode calls1 {new_encode_spy1.call_count}")
        # print(f"orig encode2 calls: {orig_encode_spy2.call_count} vs new encode calls2 {new_encode_spy2.call_count}")
        # print(f"orig encode3 calls: {orig_encode_spy3.call_count} vs new encode calls3 {new_encode_spy3.call_count}")
        print(f"TIME: {end_time - start_time}")
        
    def test_lru_encode_to_decode(self, request, cache_memo_distribution, mocker):
        problem, alg = cache_memo_distribution
        alg_spy = mocker.spy(alg, 'iterate')
        population_size, offspring_size = request.getfixturevalue("population_dim")
        cache_size =  request.getfixturevalue("max_cache_size")
        
        custom_type: CustomType = problem.types[0]
        custom_type2: CustomType = problem.types[1]
        custom_type3: CustomType = problem.types[2]
        new_encode_spy1 = mocker.spy(custom_type, "encode")
        
        start_time = perf_counter()
        alg.run(condition=ITERS)
        end_time = perf_counter()

        print(f"TIME: {end_time - start_time}")
        print(f"CACHE_SIZE {cache_size} POP/OFFSPRING: {population_size, offspring_size}"), 
        print(f"    alg ncalls: {alg_spy.call_count}")
        print(f"    encode calls1: {new_encode_spy1.call_count}")

        if cache_size > 0:
            info1 = custom_type.decode.cache_info()
            info2 = custom_type2.decode.cache_info()
            info3 = custom_type3.decode.cache_info()
            miss_perc1 = info1.misses / (info1.misses + info1.hits)
            miss_perc2 = info2.misses / (info2.misses + info2.hits)
            miss_perc3 = info3.misses / (info3.misses + info3.hits)

            print(f"CACHE INFO:")
            print(f"    cache1 ({PROBABILITIES[0]}): miss % {round(100.0*miss_perc1, 2)}, {info1}")
            print(f"    cache2 ({PROBABILITIES[1]}): miss % {round(100.0*miss_perc2, 2)}, {info2}")
            print(f"    cache3 ({PROBABILITIES[2]}): miss % {round(100.0*miss_perc3, 2)}, {info3}")
        print("\n")

def test_lru_smoke():
    bij =  _create_one_bijection()
    MonotonicDistributions(
        mappings=[bij], 
        local_mutator= DistributionShift(), 
        use_cache = True,
        max_cache_size=10
    )
    MonotonicDistributions(
        mappings=[bij], 
        local_mutator= DistributionShift(), 
        use_cache = True,
        max_cache_size=0
    )

    
def test_print_calls(single_memo_distribution: list[Solution], monkeypatch):
    problem, alg = single_memo_distribution
    evaluator = alg.evaluator
    
    custom_type: CustomType = problem.types[0]
    first_encode = custom_type.encode
    def print_first_encode(*args):
        print(f"var1 encode")
        # print(f'     input: {args[0][0]}')
        encoded = first_encode(*args)
        print(f'     encoded: {round(encoded.output_max_x, 4)}')
        return encoded
    monkeypatch.setattr(custom_type, 'encode', print_first_encode)
    
    first_decode = custom_type.decode
    def print_first_decode(*args):
        print("var1 decode")
        print(f'     input: {round(args[0].output_max_x, 4)}')
        decoded = first_decode(*args)
        print(f'     decoded {decoded[0]}')
        return decoded
    monkeypatch.setattr(custom_type, 'decode', print_first_decode)

    p_call = problem.__call__
    def print_call(*args):
        print("problem call")
        return p_call(*args)
    monkeypatch.setattr(problem, '__call__', print_call)
    
    p_eval = problem.evaluate
    def print_evaluate(*args):
        print(f"problem EVALUATE: {id(args[0]) % 100}")
        return p_eval(*args)
    monkeypatch.setattr(problem, 'evaluate', print_evaluate)
    
    alg_step= alg.step
    def print_step(*args):
        print("\n\nSTEP ---")
        return alg_step(*args)
    monkeypatch.setattr(alg, 'step', print_step)
    
    eval_all = evaluator.evaluate_all
    def print_eval_all(*args, **kwargs):
        print(f"PRE eval_all: njobs {len(args[0])}")
        result = eval_all(*args, **kwargs)
        print(f"POST  eval_all nresults {len(result), type(result[0])}")
        return result
    monkeypatch.setattr(evaluator, 'evaluate_all', print_eval_all)

    alg.run(condition=10)
    

