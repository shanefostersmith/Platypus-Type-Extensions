import numpy as np
from platypus import Solution, Problem, Real, Integer
from collections.abc import Sequence
from typing import Union, Optional
from ..utils import vectorized_to_norm, vectorized_from_norm
from ..real_methods.numba_pcx import normalized_2d_pcx
from ..integer_methods.integer_methods import multi_int_crossover

def _get_numpy_reals_and_ranges(parents: list[Solution], num_parents, problem: Problem, real_indices: Optional[Sequence]):
    parent_vars = None
    ranges = None
    
    # Put Real variables, and their ranges, in np.ndarray format
    if not real_indices:
        real_indices = []
        min_max = []
        for i, var_type in enumerate(problem.types):
            if issubclass(var_type,Real):
                min_max.append([var_type.min_value, var_type.max_value])
                real_indices.append(i)
            if not real_indices:
                return
            parent_vars = np.array(
                [[parents[i].variables[j] for j in real_indices]
                for i in range(num_parents)],
                dtype = np.float64
            )
            ranges = np.array(min_max, dtype = np.float64)
            del min_max
    else:
        ranges = np.empty((len(real_indices), 2), dtype = np.float64)
        for i,idx in enumerate(real_indices):
            var_type = problem.types[idx]
            if issubclass(var_type,Real):
                ranges[i,0] = var_type.min_value
                ranges[i,1] = var_type.max_value
            else:
                raise ValueError(f"Provide real index {idx} is not a Real (it is a {problem.types[idx]})")
        parent_vars = np.array(
            [[parents[i].variables[j] for j in real_indices]
            for i in range(num_parents)],
            dtype = np.float64
        )
    return parent_vars, ranges, real_indices
    
def real_pcx_evolve(parents: list[Solution], offspring: list[Solution], 
                    eta: Optional[float] = 0.1, 
                    zeta: Optional[float]= 0.1, 
                    real_indices: Optional[Sequence] = None,
                    original_parent_indices: Union[Sequence, int, None] = None,
                    output_dtype: type = float,
                    return_normalized_reals = False) -> Union[None,tuple[np.ndarray,np.ndarray]]:
    """
    Evolve Platypus `Real` variable types with parent-centric crossover. This method will use a Numba "no-python" version of PCX.
    
    The offspring Solution objects are updated directly.

    Args:
        parents (list[Solution]): Original parent solutions. There must be > 1 parents.
        offspring (list[Solution]): Offspring solutions. There must be > 0 offspring.
        eta (float, optional): PCX parameter. Biases reference parent value. Defaults to 0.1. Will also default to 0.1 if None.
        zeta (float, optional): PCX parameter Biases non-reference parent values (exploration). Defaults to 0.1. Will also default to 0.1 if None.
        real_indices (Sequence | None, optional): Specify which Reals to evolve with a sequence of Solution.variables indices. Defaults to None.
        
            If None, is is assumed that all Real variables in the Solutions should be updated (and will find those Real types)
            
            If not None, every provided index `i` must subscript a Platypus Real (i.e. all Solution.problem.types[i] == Real)
   
            It is slightly more efficient to provide the real_indices (even if all of variable indices that are Real type are provided)
            
        original_parent_indices (Sequence | int | None, optional): Provide one or more indices that represent what parent Solution the offspring Solution(s) were deepcopied from. Defaults to None.
        
            If None, then parents will be chosen at random to be the "reference" parent for each offspring
             
            If an integer is provided, then the same parent will be the reference for all offspring solutions
            
            If a list or numpy array, then the length of the sequence **must** be the same length as the number of offspring or length 1. 
                If length == noffspring, 'offspring[i]' Reals will evolve from PCX using reference parent 'original_parent_indices[i]'
            
            For any provided index i, must be true that i < len(parents)
        
        output_dtype (type, optional): All reals are cast to numpy float64 prior to PCX. This parameter indicates what type the final variables should be. Defaults to float.
        
        return_normalized_offspring (bool, optional): Whether not to return intermediate numpy ndarray of normalized values (usually internal parameter). Defaults to False
            If compounding variations/mutations, generally want to keep normalized values and the numpy array format.
            
            If True, return the npndarray of parent values and the np.ndarray of offspring values
    
    Raises:
        ValueError: If "real_indices" are provided and Solution.problem.types[i] is not a Platypus Real
        ZeroDivisionError: If, for any Real, it is true that Real.min_value == Real.max_value 
    """    
    
    # Ensure correct inputs
    if real_indices is not None and not real_indices:
        return
    num_parents = len(parents)
    noffspring = len(offspring)
    assert num_parents > 1 and noffspring > 0, "There must be at least 2 parents, and 1 offspring"
    if original_parent_indices is not None:
        if isinstance(original_parent_indices, Sequence):
            if len(original_parent_indices) == 1:
                original_parent_indices = int(original_parent_indices[0])
            else:
                assert len(original_parent_indices) == noffspring
        else:
            original_parent_indices = int(original_parent_indices)
        
    new_eta = np.float64(eta or 0.1)
    new_zeta = np.float64(zeta or 0.1)
    parent_vars, ranges, real_indices = _get_numpy_reals_and_ranges(parents, num_parents, parents[0].problem, real_indices)
    
    # Get offspring values
    parent_vars = vectorized_to_norm(ranges, parent_vars)
    all_offspring_vars = None
    if original_parent_indices is None:
        all_offspring_vars = normalized_2d_pcx(parent_vars, noffspring, eta, zeta) 
    elif isinstance(original_parent_indices, int):
        if original_parent_indices >= num_parents:
            raise ValueError(f"original_parent_index {original_parent_indices} is out of range")
        parent_vars[[original_parent_indices, -1]] = parent_vars[[-1, original_parent_indices]]
        all_offspring_vars = normalized_2d_pcx(parent_vars, noffspring, new_eta, new_zeta, randomize=False) 
    else:
        # More efficient to return multiple offspring at once, 
        # store count and row indices for non-unique parent indices
        counts = {} 
        for i, idx in enumerate(original_parent_indices):
            if idx >= num_parents:
                raise ValueError(f"original_parent_index {idx} is out of range")
            curr_row_indices = counts.get(idx, [])
            curr_row_indices.append(i)
            counts[idx] = curr_row_indices
        num_unique_indices = len(counts)
        if num_unique_indices == 1: # All same
            del counts
            only_idx = next(iter(counts))
            parent_vars[[only_idx, -1]] = parent_vars[[-1, only_idx]]
            all_offspring_vars = normalized_2d_pcx(parent_vars, noffspring, new_eta, new_zeta, randomize=False) 
        elif len(counts) == noffspring: # All parent indices are unique
            del counts
            all_offspring_vars = np.empty((noffspring, len(real_indices)), dtype=np.float64)
            for i, parent_idx in enumerate(original_parent_indices):
                parent_vars[[parent_idx, -1]] = parent_vars[[-1, parent_idx]]
                all_offspring_vars[i] = normalized_2d_pcx(parent_vars, 1, new_eta, new_zeta, randomize=False)[0]
        else: 
            all_offspring_vars = np.empty((noffspring, len(real_indices)), dtype=np.float64)
            for parent_idx, row_indices in counts.items():
                parent_vars[[parent_idx, -1]] = parent_vars[[-1, parent_idx]]
                temp_offspring_vars = normalized_2d_pcx(parent_vars, len(row_indices), new_eta, new_zeta, randomize=False) 
                for j, row in enumerate(row_indices):
                    all_offspring_vars[row] = temp_offspring_vars[j]
            del counts
            
    # Covert back to original scale and set new variables
    temp_type = np.float64 if not issubclass(output_dtype, np.float_) else output_dtype
    all_offspring_vars = vectorized_from_norm(ranges, all_offspring_vars, temp_type)
    for i, output in enumerate(offspring):
        for j, real_idx in enumerate(real_indices):
            output.variables[real_idx] = output_dtype(all_offspring_vars[i,j])
        output.evaluated = False

def _get_2D_numpy_integer_array(solutions: list[Solution], problem: Problem, integer_idx: int):
    
    var_type = problem.types[integer_idx]
    if not isinstance(var_type, Integer):
        raise ValueError(f"Provide real index {integer_idx} is not a Real (it is a {problem.types[integer_idx]})")
    bit_matrix = np.empty((len(solutions), var_type.nbits), np.bool_)
    for i, sol in enumerate(solutions):
        bit_matrix[i] = np.array(sol.variables[integer_idx], np.bool_)
    return bit_matrix
        
def integer_parent_crossover(parents: list[Solution], offspring: list[Solution], integer_indices = None, return_bool_arrays = False) -> Union[None,tuple[list,list]]:
    """
    Evolve Platypus Integer types with parent-centric crossover -i.e. offspring values are not used, just replaced.
    
    Offspring integers are created by mixing random sections of parent integers (per integer variable) using Numba "no-python" version of a multi-crossover
    
    The offspring Solution objects are updated directly.

    Args:
        parents (list[Solution]): Original parent solutions. There must be > 1 parents.
        offspring (list[Solution]): Offspring solutions. There must be > 0 offspring.
        integer_indices (_type_, optional): Specify which Integers to evolve with their indices a Solution.variables object. Defaults to None.

            If None, is is assumed that all Integer variables in the Solutions should be updated (and will find those Real types)
        
            If not None, every provided index `i` must subscript a Platypus Integer (i.e. all Solution.problem.types[i] == Integer )

            It is slightly more efficient to provide the integer_indices (even if all of variable indices that are Integer type are provided)
        
        return_bool_arrays (bool, optional): Whether not to return intermediate numpy ndarray of np.bool_ of parent and offspring (usually internal parameter)
            If compounding variations/mutations, generally want to keep bits in the numpy array format.
            
            If True, return the list of npndarray of parent bits and the list of np.ndarray of offspring bits
    
    Raises:
        ValueError: If "integer_indices" are provided and any Solution.problem.types[i] is not a Platypus Integer
    """      
        
    if integer_indices is not None and not integer_indices:
        return
    num_parents = len(parents)
    noffspring = len(offspring)
    assert num_parents > 1 and noffspring > 0, "There must be at least 2 parents, and 1 offspring"
    
    problem: Problem = parents[0].problem
    if integer_indices is None:
        integer_indices = [i for i, var_type in problem.types if isinstance(var_type, Integer)]
    for idx in integer_indices:
        parent_bits = _get_2D_numpy_integer_array(parents, problem, idx)
        offspring_bits = multi_int_crossover(parent_bits, noffspring)
        for i, output in enumerate(offspring):
            output.variables[idx] = offspring_bits[i].to_list()
            output.evaluated = False
        