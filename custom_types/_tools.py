
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
            other = TypeTuple(*other)
        
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

class Placeholder: 
    """`
    (Experimental)
    
    Placeholders temporarily point to Solution variables during a GlobalEvolution
        They are used indicate a variable is a reference (or read-only), and should not be deecopied immediately.
    
    Placeholder overrides `deepcopy` so that its internal reference isn't cloned. 
    When one Placeholder is used to create another, both will share the same underlying target (nested Placeholders are not created)
    
    During GlobalEvolution's evolve(), offspring Placeholders should only be replaced with brand new variables.

        *get_parent_deepcopies()* and *get_parent_deepcopy()* restore parent Solutions to there original form .
    
    **The replacement of Placeholders after an evolution or mutation should be left to the  GlobalEvolution / GlobalMutation objects**)
    """ 
         
    __slots__ = ("_temp",)
    def __init__(self, temp = None):     
        if isinstance(temp, Placeholder):
            self._temp = temp._temp
        else:
            self._temp = temp
    
    @property
    def parent_reference(self):
        """Get the parent reference this Placeholder represents. *It should be read-only*"""   
        return self._temp 
    
    def __deepcopy__(self, memo): #Avoid copy of _temp
        result = Placeholder()
        memo[id(self)] = result
        result._temp = self._temp
        return result

def _shallow_copy_solution(sol: Solution, variable_idx: int):
    """Shallow copy a Solution and its FixedLengthArray, deepcopy the variable to mutate"""
    sol_copy = copy(sol)
    sol_copy.variables = FixedLengthArray(sol.variables._size)
    sol_copy.variables._data = sol.variables._data.copy()
    sol_copy.variables[variable_idx] = deepcopy(sol.variables[variable_idx])
    return sol_copy

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
        for i, parent_idx in original_parent_choices:
            idx_map[parent_idx] = i + n_new_parents
    for offspring_idx in unaltered_offspring_choices:
        copied_from = copy_indices[offspring_idx]
        if copied_from is None:
            new_copy_indices.append(None)
            continue
        original_parents.discard(copied_from)
        new_copy_indices.append(idx_map.get(copied_from))

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
        noriginal_parents = len(original_parents)
        added_orig_parents = 0
        if n_new_parents <= max_arity:
            new_parent_choices = updated_orig_indices.copy()
        else:
            rand_indices = np.random.choice(n_new_parents, max_arity, replace = False)
            new_parent_choices = [updated_orig_indices[i] for i in rand_indices]
        if n_new_parents < min_arity or ((arity_difference > 0 and n_new_parents < min_noffspring) and noriginal_parents):
            added_orig_parents = min(max_arity - n_new_parents, noriginal_parents) if n_new_parents >= min_arity or arity_difference >= 0 else min_arity - n_new_parents
            original_parents_choices = list(original_parents) if added_orig_parents == noriginal_parents else np.random.choice(list(original_parents), added_orig_parents, replace = False)

        
        altered_offspring_choices = []
        unaltered_offspring_choices = []
        n_unaltered = 0 if not unaltered_offspring else len(unaltered_offspring)
        total_parents_added = len(new_parent_choices) + added_orig_parents
        output_offspring = max(min_noffspring, min(max_noffspring, total_parents_added - arity_difference + np.random.randint(-1, 2)))

        # Choose offspring, unaltered first
        new_offspring_to_add = 0
        if output_offspring == noffspring:
            altered_offspring_choices = updated_orig_indices
            unaltered_offspring_choices = list(unaltered_offspring)
        elif n_unaltered:
            choosen_unaltered = n_unaltered if not swaps_left else max(1,n_unaltered - swaps_left)
            if choosen_unaltered < n_unaltered and output_offspring - choosen_unaltered > n_new_parents: # ensure enough leftover offspring
                choosen_unaltered = output_offspring - n_new_parents

            unaltered_offspring_choices = list(unaltered_offspring) if choosen_unaltered >= n_unaltered else np.random.choice(list(unaltered_offspring), choosen_unaltered, replace = False)
            if choosen_unaltered < output_offspring:
                new_offspring_to_add = output_offspring - choosen_unaltered
        else:
            new_offspring_to_add = output_offspring
            
        if new_offspring_to_add:
            if new_offspring_to_add < n_new_parents:
                rand_indices = np.random.choice(n_new_parents, new_offspring_to_add)
                altered_offspring_choices = [updated_orig_indices[i] for i in rand_indices]
            else:
                altered_offspring_choices = updated_orig_indices.copy()
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

        # Choose offspring
        altered_offspring_choices = []
        unaltered_offspring_choices = []
        n_new_parents = len(updated_orig_indices)
        n_unaltered = 0 if not unaltered_offspring else len(unaltered_offspring)
        n_previously_altered = len(previously_altered)
        arity_difference = nparents - noffspring
        if min_noffspring == noffspring:
            altered_offspring_choices = updated_orig_indices.copy()
            if n_unaltered:
                unaltered_offspring_choices = list(unaltered_offspring)
        elif n_unaltered: # Unaltered first
            chosen_unaltered = n_unaltered if swaps_left <= 1 else max(1,n_unaltered - (swaps_left - 1))
            if n_unaltered < min_noffspring:
                chosen_unaltered = max(chosen_unaltered, min_noffspring - n_new_parents)
            chosen_unaltered = min(max_noffspring, chosen_unaltered)
            unaltered_offspring_choices = list(unaltered_offspring) if chosen_unaltered >= n_unaltered else np.random.choice(list(unaltered_offspring), chosen_unaltered, replace = False)
            
        nchosen = len(unaltered_offspring_choices)
        if nchosen < min_noffspring and not min_noffspring == noffspring: # Altered offspring
            additional_offspring = set(range(noffspring)).difference_update(previously_altered)
            if n_unaltered:
                additional_offspring.difference_update(unaltered_offspring)
            added_additional = 0
            if additional_offspring:
                nadditional = len(additional_offspring)
                added_additional = min(min_noffspring - nchosen, nadditional) if arity_difference > 0 else min(max_noffspring - nchosen, nadditional)
                altered_offspring_choices = list(additional_offspring) if added_additional == nadditional else np.random.choice(list(additional_offspring), added_additional, replace = False)
            if added_additional + nchosen < min_noffspring: # Previously-altered last
                from_prev = min_noffspring - (added_additional + nchosen)
                if from_prev < n_previously_altered:
                    altered_offspring_choices.extend(np.random.choice(list(previously_altered), from_prev, replace = False))
                else:
                    altered_offspring_choices.extend(previously_altered)
                    
        # Choose parents       
        total_offspring = nchosen + len(altered_offspring_choices)
        out_nparents = max(min_arity, min(total_offspring + arity_difference, max_arity))
        n_original_parents = len(original_parents)
        new_parent_choices = []
        original_parent_choices = []
        if n_previously_altered >= out_nparents: # previously altered first (the swap)
            new_parent_choices = list(previously_altered) if n_previously_altered == out_nparents else np.random.choice(list(previously_altered), out_nparents, replace = False)
        elif not n_original_parents or n_new_parents >= out_nparents: # other updated
            new_parent_choices = list(previously_altered)
            updated_left = set(updated_orig_indices).difference_update(previously_altered)
            if updated_left:
                new_parent_choices.extend(np.random.choice(list(updated_left), min(len(updated_left), out_nparents - n_previously_altered), replace = False))
        if len(new_parent_choices) < out_nparents: # uncopied/unused parents
            original_parent_choices = np.random.choice(list(original_parent_choices), out_nparents - len(new_parent_choices), replace = False)
        return new_parent_choices, original_parent_choices, altered_offspring_choices, unaltered_offspring_choices

def _previous_selection(updated_orig_indices: list, original_parents: set, unaltered_offspring: set, previously_altered: set,
                        nparents: int, noffspring: int, swaps_left: int, min_arity, max_arity, min_noffspring, max_noffspring):
        
        n_previously_altered = len(previously_altered)
        n_unaltered = 0 if not unaltered_offspring else len(unaltered_offspring)
        out_noffspring = max(min_noffspring, min(n_previously_altered, max_noffspring))
        unaltered_offspring_choices = []
        altered_offspring_choices = None
        
        # Choose offspring
        if out_noffspring == noffspring:
            altered_offspring_choices = updated_orig_indices.copy()
            if n_unaltered:
                unaltered_offspring_choices = list(unaltered_offspring)
        else: # previously-altered first
            altered_offspring_choices = list(previously_altered) if n_previously_altered <= out_noffspring else np.random.choice(list(previously_altered), out_noffspring, replace = False)
            if n_previously_altered < out_noffspring: 
                offspring_left = set(updated_orig_indices).difference_update(previously_altered)
                n_updated_left = len(offspring_left)
                from_ua = 0
                if n_unaltered: # unaltered next
                    from_ua = max(1, n_unaltered - swaps_left)
                    if out_noffspring - n_previously_altered - n_updated_left > from_ua:
                        from_ua = min(n_unaltered, out_noffspring - n_previously_altered - n_updated_left)
                    unaltered_offspring_choices = list(unaltered_offspring) if from_ua == n_unaltered else np.random.choice(list(unaltered_offspring), from_ua, replace = False)
                if from_ua + n_previously_altered < out_noffspring: # other altered
                    to_choose = out_noffspring - (from_ua + n_previously_altered)
                    if to_choose < offspring_left:
                        altered_offspring_choices.extend(list[offspring_left][:to_choose])
                    else:
                        altered_offspring_choices.extend(list[offspring_left])
                        
        n_new_parents = len(updated_orig_indices)
        n_original_parents = len(original_parents)
        out_nparents = max(min_arity, min(out_noffspring + (nparents - noffspring), max_arity))
        new_parent_choices = []
        original_parent_choices = []
        if not n_original_parents or n_new_parents >= out_noffspring:
            new_parent_choices = np.random.choice(updated_orig_indices, max(n_new_parents , out_nparents), replace = False)
        else:
            new_parent_choices = updated_orig_indices.copy()
            from_orig= out_nparents - n_new_parents
            original_parent_choices = np.random.choice(list(original_parents), from_orig, replace = False) if from_orig < n_original_parents else list(original_parents)
        return new_parent_choices, original_parent_choices, altered_offspring_choices, unaltered_offspring_choices
    

# def _lazycopy_evolve(self: GlobalEvolution, parents: list[Solution]):
#     result = None
#     if not self._offspring_input: 
#         for idx in self._avoid_copy_indices:
#             for p in parents:
#                 p.variables[idx] = Placeholder(p.variables[idx])
                
#         result = self._evolve(parents)
        
#         if not self._did_replacement: # Ensure parents are restored to original post-evolve
#             for idx in self._avoid_copy_indices:
#                 for par in parents:
#                     ph = par.variables[idx]
#                     if isinstance(ph,Placeholder):
#                         par.variables[idx] = ph.get_parent_reference()
#         else:
#             self._did_replacement = False
            
#     else: # If offspring input, no deepcopying
#         result = self._evolve(parents)
       
#     if not self._next_operator: # Replace offspring with deepcopy of parent (or next operator will handle it)
#         for idx in self._avoid_copy_indices:
#             for offspring in result:
#                 ph = offspring.variables[idx]
#                 if isinstance(ph,Placeholder):
#                     offspring = deepcopy(ph.get_parent_reference())
#     return result

# def _lazycopy_mutate(self: GlobalMutation, parent: Solution):
#     result = None
#     if not self._offspring_input:
#         for idx in self._avoid_copy_indices:
#             parent.variables[idx] = Placeholder(parent.variables[idx])

#         result = self._mutate(parent)
        
#         if not self._did_replacement:
#             for idx in self._avoid_copy_indices:
#                 ph = parent.variables[idx]
#                 if isinstance(ph,Placeholder):
#                     parent.variables[idx] = ph.get_parent_reference()
#     else:
#         result = self._mutate(parent)

#     # Replace all parent and offspring Placeholders still present
#     if not self._next_operator:
#         for idx in self._avoid_copy_indices:
#             for offspring in result:
#                 ph = offspring.variables[idx]
#                 if isinstance(ph,Placeholder):
#                     offspring = deepcopy(ph.get_parent_reference())

#     return result

# class GlobalMutation(Mutation):
#     """
#     An interface representing a global mutation. Streamlines mutation of Solution variables when various types exist.
    
#     Subclasses must implement `mutate()`


#     """   
#     def __init__(self, 
#                  problem: Problem, 
#                  ignore_generics = True):
        
#         super(GlobalMutation, self).__init__()
#         to_mutate = []
#         avoid_copy_indices = []
        
#         # Determine which types in the Problem are able to mutate
#         # Determine CustomTypes have variables not to directly copy
#         for i, var_type in enumerate(problem.types):
#             if isinstance(var_type, CustomType) and var_type.do_mutation: 
#                 to_mutate.append[i]
#                 if var_type._lazy_copy: 
#                     avoid_copy_indices.append[i]     
#             elif not issubclass(var_type, CustomType) and not ignore_generics:
#                 to_mutate.append[i]
       
#         if len(to_mutate) == 0:
#             raise ValueError(
#                 "None of the variable types in the problem can be mutated. If mutating a CustomType, ensure it's 'do_mutation' attribute is set to True. Or, if only mutating Platypus generic, set to False")

#         if len(avoid_copy_indices) > 0:
#             avoid_copy_indices = frozenset(avoid_copy_indices)
#             self._mutate = self.mutate
#         else:
#             avoid_copy_indices = None
            
#         self._avoid_copy_indices = avoid_copy_indices
#         self._vars_to_mutate = range(problem.nvars) if len(to_mutate) == problem.nvars else tuple(to_mutate)
        
#         # Internal, for compound operators / lazy copying
#         self._did_replacement = False
#         self._offspring_input = False 
#         self._next_operator = False
    
#     def __setattr__(self, name, value):
#         if name in ('_mutate, _vars_to_mutate'):
#             raise AttributeError(f"Cannot reassign read-only attribute {name!r}")
#         super().__setattr__(name, value)
    

#     # def replace_all_parent_placeholders(self, parent: Solution):
#     #     """
#     #     Restore a parent Solution to its original form.
        
#     #     All variables for are checked PlaceholderPairs. If a variable does have PlaceholderPair present, replace it with the original (encoded) variable.
            
#     #     (If applicable, should be called after a deepcopy is created. This input should be the original Solution, not the deepcopy Solution.)
#     #     """        
#     #     if not self._avoid_copy_indices:
#     #         return parent
#     #     for i in self._avoid_copy_indices:
#     #         placeholder = self.get_encoded_placeholder(parent)
#     #         if isinstance(placeholder, Placeholder):
#     #             parent.variables[i] = placeholder.get_parent_reference()
    
#     def get_parent_copy(self, parent: Solution) -> Solution:
#         """
#         Get a deepcopy of a parent `Solution`. 
        
#         If applicable, all PlaceholderPairs will be replaced with the original encoded data.
        
#         """
#         out = parent if self._offspring_input else deepcopy(parent) 
#         return out
#         # if self._avoid_copy_indices:
#         #     self.replace_all_parent_placeholders(parent)
#         # return out
        
#     def get_variable_indices_to_mutate(self) -> Tuple[int, ...]:
#         """
#         Returns:
#             Tuple[int, ...]: A tuple of `Solution.variables` indices where mutation is valid 
#         """        
#         return  (*self._vars_to_mutate,) if isinstance(self._vars_to_mutate, range) else self._vars_to_mutate
    
#     def generate_variables_to_mutate(self, child: Solution) -> Generator[object, type, int]:
#         """`
        
#         Generate variables of a child `Solution` that can be mutated. Assumes that child is a deepcopied parent Solutions
        
#         Will only include CustomTypes where *CustomType.do_mutation* is True
        
#         Will only include Platypus Types if *self.ignore_generics* is False
      
#         Args:
#             child (Solution): A deepcopy of a parent `Solution`

#         Yields:
#             Generator[Any, type, int]: a tuple containing:
            
#             - A child Solution's variable  (may be a Placeholder if LazyProblem is used)

#             - The CustomType or Type subclass the variable is derived from
                
#             - The index of the variable in the `child.variables` 
#         """ 
#         problem: Problem = child.problem
#         problem_types = problem.types
#         # for i in self._vars_to_mutate:
#         #     yield self.get_encoded_placeholder(child, i), problem_types[i], i


# class GlobalGAOperator(Variator):
#     """
#     A GAOperator that incorporates the lazy copying functionality of `GlobalEvolution` and `GlobalMutation`
    
#     If lazy copying is enabled, then this class will manage *construct_only* variables between `GlobalEvolution` and `GlobalMutation`
    
#     Deepcopying of Solutions only occurs at the start of `GlobalEvolution`
#          `GlobalMutation` is flagged to recognize that an input Solution is an offspring (not a parent)
    
#     """
#     def __init__(
#         self, 
#         global_evolution: GlobalEvolution, 
#         global_mutation: GlobalMutation):
        
#         global_mutation._offspring_input = True
        
#         # Ensure GlobalEvolution is aware of 'construct_only' variable that are not evolved
#         if global_mutation._avoid_copy_indices: 
#             if global_evolution._avoid_copy_indices is None:
#                 global_evolution._avoid_copy_indices = global_mutation._avoid_copy_indices
#             else:
#                 combined_avoid = global_evolution._avoid_copy_indices | global_mutation._avoid_copy_indices
#                 global_evolution._avoid_copy_indices = combined_avoid
        
#         self.global_evolution = global_evolution
#         self.global_mutation = global_mutation

#     def evolve(self, parents):
#         return list(map(self.global_mutation.evolve, self.global_evolution.evolve(parents)))