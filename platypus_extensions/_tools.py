
import numpy as np
from platypus import Solution, FixedLengthArray
from copy import copy, deepcopy
from collections.abc import Sequence
from collections import namedtuple


class TypeTuple(tuple):
    """Tuple for types, where __contains__ also does subclass equality"""
    def __new__(cls, *types):
        return super().__new__(cls, types)
    
    def __contains__(self, other):
        cls_ = other if isinstance(other, type) else type(other)
        return any(issubclass(cls_, t) for t in self)
    
    def __add__(self, other):
        return TypeTuple(*super().__add__(other))
    
    def __getnewargs_ex__(self):
        return tuple(self), {}
    
    def __and__(self, other):
        """Return a TypeTuple of the most specific types common to self and the other TypeTuple."""
        if not isinstance(other, TypeTuple):
            other = TypeTuple(*other) if hasattr(other, '__iter__') else TypeTuple(other)
        
        candidates = []
        for t in self:
            for u in other:
                if issubclass(t, u):
                    candidates.append(t)
                elif issubclass(u, t):
                    candidates.append(u)

        # Remove any candidate that has a *more* specific type in candidates
        pruned = []
        for c in candidates:
            if not any(d is not c and issubclass(d, c) for d in candidates):
                pruned.append(c)
    
        return TypeTuple(*set(pruned))
        
    
    def __str__(self):
        return  ", ".join(t.__name__ for t in self)

def _deep_eq(d1, d2):
    if type(d1) != type(d2):
        return False
    if isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
        return d1.shape == d2.shape and np.all(d1 == d2)
    if isinstance(d1, Sequence) and isinstance(d2, Sequence):
        if len(d1) != len(d2):
            return False
        elif isinstance(d1, str):
            return d1 == d2
        return all(_deep_eq(x, y) for x, y in zip(d1, d2))
    return d1 == d2

def _single_memo_encode(self, decoded):
    """For remembering encodings from decoding"""
    if self._enc_temp is None:
        return self._mem_encode(decoded)
    else:
        encoding = self._enc_temp[0]
        # print(f"     same?: {_deep_eq(decoded, self._enc_temp[1])}")
        self._enc_temp = None
        return encoding

def _single_memo_decode(self, encoded):
    """For remembering encodings from decoding"""
    decoded = self._mem_decode(encoded)
    self._enc_temp = (encoded, decoded)
    return decoded

def _shallow_copy_solution(sol: Solution, variable_idx: int):
    """Shallow copy a Solution and its FixedLengthArray, deepcopy the variable to mutate"""
    sol_copy = copy(sol)
    sol_copy.variables = FixedLengthArray(sol.variables._size)
    sol_copy.variables._data = sol.variables._data.copy()
    sol_copy.variables[variable_idx] = deepcopy(sol.variables[variable_idx])
    return sol_copy

def _copy_indices_and_deepcopy(offspring_solutions: list[Solution], variable_index: int,
                                   new_parent_choices: list, altered_offspring_choices: list):
    """
    Returns: tuple(list, list, list, dict)
        - 1. list of offspring_solutions (potential shallow_copies + deepcopied variable)
        - 2. New copy_indices
        - 3. list of "replacements" tuples -> (indices of first list, index of 'offspring_solutions')
    """        
    # no overlap
    if not altered_offspring_choices: # no overlap
        return [], [], []

    idx_map = {}
    copy_indices = []
    output_altered_solutions = []
    to_replace = []
    for i, offspring_idx in enumerate(new_parent_choices):
        idx_map[offspring_idx] = i
    for i, offspring_idx in enumerate(altered_offspring_choices):
        corresponding_idx = idx_map.get(offspring_idx)
        if corresponding_idx is None:
            copy_indices.append(None)
            output_altered_solutions.append(offspring_solutions[offspring_idx])
            continue
        copy_indices.append(corresponding_idx)
        to_replace.append((i, offspring_idx))
        output_altered_solutions.append(_shallow_copy_solution(offspring_solutions[offspring_idx], variable_index))

    return output_altered_solutions, copy_indices, to_replace
    
def _unaltered_overlap(
    original_parents: set, 
    copy_indices: list, 
    unaltered_offspring_choices: list, 
    original_parent_choices: list,
    new_copy_indices: list, 
    n_new_parents: int):
    """
    (for LocalCompoundOperator)
    Add to new copy indices. Determine if any unaltered offspring choices result in removing original parents"""
    
    if not unaltered_offspring_choices:
        return
    
    idx_map = {}
    if original_parent_choices:
        for i, parent_idx in enumerate(original_parent_choices):
            idx_map[parent_idx] = i + n_new_parents
            
    for offspring_idx in unaltered_offspring_choices:
        copied_from = copy_indices[offspring_idx]
        if copied_from is None:
            new_copy_indices.append(None)
            continue
        original_parents.discard(copied_from)
        new_copy_indices.append(idx_map.get(copied_from))

def _variable_replacement(offspring_solutions: list[Solution], new_offspring: list[Solution], 
                              replacement_indices: list[tuple[int,int]], variable_index: int):
    """Update variable and evaluated flag of original offspring solution, for all replacements"""
    if not replacement_indices:
        return
    for new_idx, offspring_idx in replacement_indices:
        shallow_copy_sol = new_offspring[new_idx]
        orig_offspring = offspring_solutions[offspring_idx]
        orig_offspring.variables[variable_index] = shallow_copy_sol.variables[variable_index]
        orig_offspring.evaluated = shallow_copy_sol.evaluated

def _rand_selection(updated_orig_indices: list, original_parents: set, unaltered_offspring: set,
                    nparents: int, noffspring: int, swaps_left: int,
                    min_arity, max_arity, min_noffspring, max_noffspring):
        """
        (for LocalCompoundOperator)
        Returns: tuple(list, list, list, list)
            - indices of altered offspring (as parents)
            - indices of original_parents 
            - indices of altered offspring (as offspring)
            - indices of unaltered offspring
        """
        
        arity_difference = nparents - noffspring

        # Choose parent solutions, updated first
        new_parent_choices = None
        original_parents_choices = None
        n_new_parents = len(updated_orig_indices)
        n_original_parents = len(original_parents)
        added_orig_parents = 0
        
        new_parent_choices = updated_orig_indices.copy() if n_new_parents <= max_arity else np.random.choice(updated_orig_indices, max_arity, replace = False)
        if n_new_parents < min_arity or ((arity_difference > 0 and n_new_parents <= min_noffspring) and n_original_parents):
            added_orig_parents = min(max_arity - n_new_parents, n_original_parents) if n_new_parents >= min_arity or arity_difference >= 0 else min_arity - n_new_parents
            original_parents_choices = list(original_parents) if added_orig_parents == n_original_parents else np.random.choice(list(original_parents), added_orig_parents, replace = False).tolist()

        altered_offspring_choices = []
        unaltered_offspring_choices = []
        n_unaltered = 0 if not unaltered_offspring else len(unaltered_offspring)
        total_parents_added = len(new_parent_choices) + added_orig_parents
        output_offspring = max(min_noffspring, min(max_noffspring, total_parents_added - arity_difference + np.random.randint(-1, 2)))

        # Choose offspring, unaltered first
        new_offspring_to_add = 0
        if output_offspring == noffspring:
            altered_offspring_choices = updated_orig_indices
            unaltered_offspring_choices = [] if not unaltered_offspring else list(unaltered_offspring)
        elif n_unaltered:
            choosen_unaltered = n_unaltered if not swaps_left else max(1,n_unaltered - swaps_left)
            if choosen_unaltered < n_unaltered and output_offspring - choosen_unaltered > n_new_parents: # ensure enough leftover offspring
                choosen_unaltered = output_offspring - n_new_parents

            unaltered_offspring_choices = list(unaltered_offspring) if choosen_unaltered >= n_unaltered else np.random.choice(list(unaltered_offspring), choosen_unaltered, replace = False).tolist()
            if choosen_unaltered < output_offspring:
                new_offspring_to_add = output_offspring - choosen_unaltered
        else:
            new_offspring_to_add = output_offspring
            
        if new_offspring_to_add:
            altered_offspring_choices = np.random.choice(updated_orig_indices, new_offspring_to_add, replace = False).tolist() if new_offspring_to_add < n_new_parents else updated_orig_indices.copy()
                
        return new_parent_choices, original_parents_choices, altered_offspring_choices, unaltered_offspring_choices

def _swap_selection(updated_orig_indices: list, original_parents: set, unaltered_offspring: set, previously_altered: set,
                    nparents: int, noffspring: int, swaps_left: int, min_arity, max_arity, min_noffspring, max_noffspring):
        """
        (for LocalCompoundOperator)
        Returns: tuple(list, list, list, list)
            - indices of altered offspring (as parents)
            - indices of original_parents 
            - indices of altered offspring (as offspring)
            - indices of unaltered offspring
        """
        n_new_parents = len(updated_orig_indices)
        n_unaltered = 0 if not unaltered_offspring else len(unaltered_offspring)
        n_previously_altered = len(previously_altered)
        n_original_parents = len(original_parents)
        arity_difference = nparents - noffspring
        
        # Choose offspring
        altered_offspring_choices = []
        unaltered_offspring_choices = []
        unaltered_list = list(unaltered_offspring)
        n_unaltered_chosen = 0
        if min_noffspring == noffspring:
            altered_offspring_choices = updated_orig_indices.copy()
            if n_unaltered:
                unaltered_offspring_choices = unaltered_list
                n_unaltered_chosen = n_unaltered
                
        elif n_unaltered: # Unaltered first
            n_unaltered_chosen= n_unaltered if swaps_left == 0 else max(1,n_unaltered - (swaps_left - 1))
            if n_unaltered < min_noffspring:
                n_unaltered_chosen = max(n_unaltered_chosen, min_noffspring - n_new_parents)
                
            n_unaltered_chosen = min(max_noffspring, n_unaltered_chosen)
            unaltered_offspring_choices = unaltered_list if n_unaltered_chosen >= n_unaltered else np.random.choice(unaltered_list, n_unaltered_chosen, replace = False).tolist()
        
        previously_altered_list = list(previously_altered)
        if n_unaltered_chosen < min_noffspring and not min_noffspring == noffspring: # Altered offspring
            additional_offspring = np.setdiff1d(np.arange(noffspring), previously_altered_list, assume_unique=True)
            if n_unaltered:
                additional_offspring = np.setdiff1d(additional_offspring, unaltered_list, assume_unique=True)
            additional_offspring = additional_offspring.tolist()
                
            added_additional = 0
            if additional_offspring:
                nadditional = len(additional_offspring)
                added_additional = min(min_noffspring - n_unaltered_chosen, nadditional) if arity_difference > 0 else min(max_noffspring - n_unaltered_chosen, nadditional)
                altered_offspring_choices = list(additional_offspring) if added_additional == nadditional else np.random.choice(additional_offspring, added_additional, replace = False).tolist()
                
            if added_additional + n_unaltered_chosen < min_noffspring: # previously-altered offspring
                from_prev = min_noffspring - (added_additional + n_unaltered_chosen)
                if from_prev < n_previously_altered:
                    altered_offspring_choices.extend(np.random.choice(previously_altered_list, from_prev, replace = False))
                else:
                    altered_offspring_choices.extend(previously_altered_list)

        # Choose parents       
        total_offspring = n_unaltered_chosen + len(altered_offspring_choices)
        out_nparents = max(min_arity, min(total_offspring + arity_difference, max_arity))
        nparents_chosen = 0
        new_parent_choices = None
        original_parent_choices = []
        if n_previously_altered >= out_nparents: # previously altered first (the swap)
            new_parent_choices = previously_altered_list if n_previously_altered == out_nparents else np.random.choice(previously_altered_list, out_nparents, replace = False).tolist()
            nparents_chosen = out_nparents
        else:
            new_parent_choices = previously_altered_list
            nparents_chosen += n_previously_altered
            
        if n_original_parents + n_previously_altered < out_nparents: # Other altered offspring
            updated_left = updated_orig_indices if not nparents_chosen else np.setdiff1d(updated_orig_indices, new_parent_choices, assume_unique=True).tolist()
            if updated_left:
                max_addition = min(len(updated_left), out_nparents - n_previously_altered - n_original_parents)
                new_parent_choices.extend(np.random.choice(updated_left, max_addition, replace = False))
                nparents_chosen += max_addition 
                
        if nparents_chosen < out_nparents: # uncopied/unused parents
            original_parent_choices = np.random.choice(list(original_parents), out_nparents - nparents_chosen, replace = False).tolist()
            
        assert len(new_parent_choices) + len(original_parent_choices) == out_nparents
        assert n_unaltered_chosen + len(altered_offspring_choices) >= min_noffspring
        return new_parent_choices, original_parent_choices, altered_offspring_choices, unaltered_offspring_choices

def _previous_selection(updated_orig_indices: list, original_parents: set, unaltered_offspring: set, previously_altered: set,
                        nparents: int, noffspring: int, swaps_left: int, min_arity, max_arity, min_noffspring, max_noffspring):
        
        n_previously_altered = len(previously_altered)
        n_unaltered = 0 if not unaltered_offspring else len(unaltered_offspring)
        n_new_parents = len(updated_orig_indices)
        n_original_parents = len(original_parents)
        
        out_noffspring = max(min_noffspring, min(n_previously_altered, max_noffspring))
        unaltered_offspring_choices = []
        altered_offspring_choices = None
        
        # Choose offspring
        unaltered_list = list(unaltered_offspring)
        if out_noffspring == noffspring:
            altered_offspring_choices = updated_orig_indices.copy()
            if n_unaltered:
                unaltered_offspring_choices = unaltered_list
        else: 
            # previously-altered first
            altered_offspring_choices = list(previously_altered) if n_previously_altered <= out_noffspring else np.random.choice(list(previously_altered), out_noffspring, replace = False).tolist()
            if n_previously_altered < out_noffspring: 
                n_updated_left = max(0, noffspring - n_previously_altered - n_unaltered)
                
                total_chosen = n_previously_altered
                if n_unaltered > swaps_left or (n_unaltered > 0 and n_updated_left + n_previously_altered < out_noffspring): # unaltered next
                    unaltered_chosen = max(1, n_unaltered - swaps_left)
                    if out_noffspring - n_previously_altered - n_updated_left > unaltered_chosen:
                        unaltered_chosen = min(n_unaltered, out_noffspring - n_previously_altered - n_updated_left)
                    unaltered_offspring_choices = unaltered_list  if unaltered_chosen == n_unaltered else np.random.choice(unaltered_list, unaltered_chosen, replace = False).tolist()
                    total_chosen += unaltered_chosen
                    
                if total_chosen < out_noffspring: # other altered
                    updated_remaining = np.setdiff1d(updated_orig_indices, altered_offspring_choices, assume_unique=True)
                    to_choose = out_noffspring - n_updated_left
                    if to_choose < n_updated_left:
                        altered_offspring_choices.extend(updated_remaining[:to_choose])
                    else:
                        altered_offspring_choices.extend(updated_remaining)
            
            assert len(altered_offspring_choices) + len(unaltered_offspring_choices) == out_noffspring
                        
        out_nparents = max(min_arity, min(out_noffspring + (nparents - noffspring), max_arity))
        new_parent_choices = []
        original_parent_choices = []
        if n_new_parents >= out_nparents:
            new_parent_choices = updated_orig_indices.copy() if n_new_parents == out_nparents else np.random.choice(updated_orig_indices, out_nparents, replace = False).tolist()
        else:
            new_parent_choices = updated_orig_indices.copy()
            from_orig= out_nparents - n_new_parents
            original_parent_choices = np.random.choice(list(original_parents), from_orig, replace = False).tolist() if from_orig < n_original_parents else list(original_parents)
        
        # assert len(new_parent_choices) + len(original_parent_choices) == out_nparents, f"num_new {len(new_parent_choices)} out of {n_new_parents}, num_og {len(original_parent_choices)}, out of {n_original_parents}"
        return new_parent_choices, original_parent_choices, altered_offspring_choices, unaltered_offspring_choices

def _first_variator_subset_case(
    variator, custom_type,
    parent_solutions, offspring_solutions, 
    copy_indices, variable_index,
    out_nparents, out_noffspring,
    nparents, noffspring, **kwargs
):
    """Case when out_nparents < nparents and out_noffspring < noffspring.
    Choose offspring and parent solution, attempt to maintain diversity of parent solutions (given copy indices)

    Args:
        variator (LocalVariator)
        custom_type (CustomType)
        parent_solutions (list[Solution])
        offspring_solutions (list[Solution])
        copy_indices (list)
        variable_index (int): _description_
        out_nparents (int): actual number of parent solutions to choose
        out_noffspring (_type_): actual number of offspring solutions to choose
        nparents (int)
        noffspring (int)

    Returns:
        set: set of offspring indices that were chosen / evolved
    """
    
    out_parent_indices = []
    out_offspring_indices = []
    unlabeled_offspring = []
    duplicate_indices = [] 
    new_copy_indices = []
    parent_map = {} # parent_idx -> idx in out_parent_indices
    added = 0
    for i, parent_idx in enumerate(copy_indices):
        if parent_idx is None:
            unlabeled_offspring.append(i)
            continue
        if parent_idx in parent_map:
            duplicate_indices.append(i)
            continue
        
        out_parent_indices.append(parent_idx)
        out_offspring_indices.append(i)
        new_copy_indices.append(added)
        parent_map[parent_idx] = added
        added += 1
        if added == out_noffspring or added == out_nparents:
            break
    
    if len(unlabeled_offspring) == noffspring: # All offspring are unlabeled
        variator.evolve(
            custom_type, 
            parent_solutions[:out_nparents].copy(), offspring_solutions[:out_noffspring].copy(), 
            variable_index, copy_indices = [None for _ in range(out_noffspring)], 
            **kwargs)
        return set(range(out_noffspring))
    
    offspring_to_add = out_noffspring - added
    parents_to_add = out_nparents - added
    if not parents_to_add and not offspring_to_add:
        new_parents = [parent_solutions[i] for i in out_parent_indices]
        new_offspring = [offspring_solutions[i] for i in out_offspring_indices]
        variator.evolve(custom_type, new_parents, new_offspring, variable_index, new_copy_indices, **kwargs)
        return set(out_offspring_indices)
    
    if not parents_to_add and offspring_to_add:
        for offspring_idx in duplicate_indices: # offspring whose parent is already added
            out_offspring_indices.append(offspring_idx)
            new_copy_indices.append(parent_map[copy_indices[offspring_idx]])
            offspring_to_add -= 1
            if not offspring_to_add:
                break

        if offspring_to_add and unlabeled_offspring: # add unlabeled offspring
            for offspring_idx in unlabeled_offspring:
                out_offspring_indices.append(offspring_idx)
                new_copy_indices.append(None)
                offspring_to_add -= 1
                if not offspring_to_add:
                    break
                
        if offspring_to_add: # add potentially unseen
            offspring_left = np.setdiff1d(np.arange(noffspring), out_offspring_indices, assume_unique=True)
            if offspring_left > offspring_to_add:
                out_offspring_indices.extend(offspring_left[:offspring_to_add])
            else:
                out_offspring_indices.extend(offspring_left)
            new_copy_indices.extend(None for _ in offspring_to_add)
        
    else:
        if parents_to_add:
            uncopied = np.setdiff1d(np.arange(nparents), list(parent_map.keys()), assume_unique=True)
            out_parent_indices.extend(i for i in uncopied[:parents_to_add])
            
        if offspring_to_add: # Went through loop without breaking
            if unlabeled_offspring:
                unlabeled_to_add = min(len(unlabeled_offspring), offspring_to_add)
                out_offspring_indices.extend(unlabeled_offspring[:unlabeled_to_add])
                offspring_to_add -= unlabeled_to_add
                new_copy_indices.extend(None for _ in range(unlabeled_to_add))
                
            for offspring_idx in duplicate_indices:
                out_offspring_indices.append(offspring_idx)
                new_copy_indices.append(parent_map[copy_indices[offspring_idx]])
                offspring_to_add -= 1
                if not offspring_to_add:
                    break
                
    new_offspring = [offspring_solutions[i] for i in out_offspring_indices]    
    new_parents = [parent_solutions[i] for i in out_parent_indices]      
    variator.evolve(custom_type, new_parents, new_offspring, variable_index, new_copy_indices, **kwargs)
    return set(out_offspring_indices)
        
