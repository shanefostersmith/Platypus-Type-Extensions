from platypus_extensions.core import CustomType, LocalVariator, LocalMutator, PlatypusType
from platypus import Variator, Mutation, Solution, Problem, GAOperator
from typing import Literal, Generator
from copy import deepcopy

class GlobalEvolution2(Variator):
    """
    (For separation GlobalEvolution and GlobalMutation
    would require the CustomType `do_evolution` and `do_mutation` to be Value types from the multiprocessing library
    or some other synchronization-safe object)
    
    An interface representing a global evolution. Streamlines evolution/crossover of Solution variables when various types exist.
    
    Subclasses must implement `evolve()`
    
    This class handles:
        - Defining an arity and number of offspring to produce
        - Deepcopying parent solutions to create offspring solutions
        - Directing and defining any non-uniform mutation, if applicable
        - Determining if CustomType variable can evolve or mutate given the *do_evolution* and *do_mutation* flags.
        - Calling LocalVariators of CustomType's and returning the offspring Solutions
    
    """    
    def __init__(self, 
                 arity: int,
                 noffspring: int,
                 ignore_generics = True):
        
        assert arity >= 1 and noffspring >= 1, "Must provide at least 1 parent and 1 offspring"
        super(GlobalEvolution2, self).__init__(arity)
        self.noffspring = noffspring
        self._ignore_generics = ignore_generics
        self._indices_by_type = None
        self._override_mutation = False

    def get_parent_deepcopies(
        self,
        parents: list[Solution], 
        copy_indices) -> list[Solution]:      
        """ 
        Create a one or more deepcopies of a parent Solutions (ie. offspring Solutions)
   
        Args:
            parent (Solution, list[Solution]): A parent solution or a list of parent Solutions
            copy_indices (int, list[int], Literal['all'], optional): One or more indices in *parents* to deepcopy. Defaults to 'all'
                - If *parent_indices* is 'all', each parent will be deepcopied once.
  
        Returns:
            List[Solution]: A list of deepcopied Solutions 
        """

        out = None
        if isinstance(copy_indices, int):
            out =[deepcopy(parents[copy_indices])]
        else:
            if copy_indices == 'all':
                copy_indices = range(len(parents))
            out = [deepcopy(parents[i]) for i in copy_indices]
        
        return out
    
    def is_mutatable_type(self, problem_type) -> bool:
        """Check if a CustomType or Platypus Type can be evolved given it's """
        return (isinstance(problem_type, CustomType) and problem_type.can_evolve) or (not isinstance(problem_type, CustomType) and not self._ignore_generics)

    
    def generate_types_to_evolve(self, problem: Problem):
        """
        Generate CustomType objects and their variable_index 
        
        Will only include CustomTypes where the *custom.can_evolve* property is True (set by *do_evolution* and *do_mutation*)
            
        Will only include Platypus Types if 'ignore_generics' is False
        
        Args:
            problem: The Problem attribute of the Solution objects

        Yields:
            Generator[tuple[CustomType | PlatypusType, int, LocalVariator | None]]:
            - A CustomType or Platypus Type of the object
            - variable_index: The index of the variable in the `offspring.variables` FixedLengthArray
        """  
        problem_types = problem.types
        override_flag = self._override_flag
        # read do_mutation / do_evolution before
        if not override_flag:
            for i, var_type in enumerate(problem_types):
                # lock / unlock mananger
                if self.is_mutatable_type(var_type):
                    yield var_type, i
        else:
            """ 
            For multiprocessing, 
                before loop / enter: 
                    iterate through vars, lock writing, read do_evolution / do_mutation and store -> unlock
                in loop: 
                    override do_evolution / do_mutation if applicable, 
                    set writing lock during one evolution / mutation -> unlock 
                could be done w/ context manager or context manager + ExitStack
                basic version would execute the LocalVariator as well
            """
            # make ContextManager
            override_flag = self._override_flag
            for i, var_type in enumerate(problem_types):
                override = False
                try: 
                    if getattr(var_type, override_flag, False):
                        override = True
                        setattr(var_type, override_flag, False)
                    if self.is_mutatable_type(var_type):
                        yield var_type, i 
                except:
                    raise
                finally:
                    if override:
                        setattr(var_type, override_flag, True)
              
    @property
    def _override_flag(self):
        return None if not self._override_mutation else 'do_mutation'
    
    def _store_problem_types(self, problem: Problem):
        """Store indices of a Problem's variable types, only does not recalculate if it exists."""
        if self._indices_by_type is not None:
            return
        indices_by_type = {}
        problem_types = problem.types
        for i, var_type in enumerate(problem_types):
            if not isinstance(var_type, CustomType) and self._ignore_generics:
                continue
            indices_by_type.setdefault(type(var_type), []).append(i)
    
        self._indices_by_type = tuple(tuple(indices) for indices in indices_by_type.values())

class GlobalMutation(Mutation):
    def __init__(self, ignore_generics = True):
        self._ignore_generics = ignore_generics
        self._indices_by_type = None
        super(GlobalMutation, self).__init__()
    
    def get_parent_deepcopy(parent: Solution):
        return deepcopy(parent)
    
    _store_problem_types = GlobalEvolution2._store_problem_types
    is_mutatable_type = GlobalEvolution2.is_mutatable_type
    generate_types_to_evolve = GlobalEvolution2.generate_types_to_evolve

    @property
    def _override_flag(self):
        return 'do_evolution'

def create_GlobalGAOperator(
    global_evolution: GlobalEvolution2, 
    global_mutation: GlobalMutation):
    
    global_evolution._override_mutation = True
    return GAOperator(global_evolution, global_mutation)

class Placeholder: 
    """`
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


#     def get_parent_copy(self, parent: Solution) -> Solution:
#         """
#         Get a deepcopy of a parent `Solution`. 
        
#         If applicable, all PlaceholderPairs will be replaced with the original encoded data.
        
#         """
#         out = parent if self._offspring_input else deepcopy(parent) 
#         return out
#         if self._avoid_copy_indices:
#             self.replace_all_parent_placeholders(parent)
#         return out

#     def replace_all_parent_placeholders(self, parent: Solution):
#          """
#          Restore a parent Solution to its original form.
        
#          All variables for are checked Placeholder. If a variable does have Placeholder present, replace it with the original (encoded) variable.
            
#          (If applicable, should be called after a deepcopy is created. This input should be the original Solution, not the deepcopy Solution.)
#           """        
#          if not self._avoid_copy_indices:
#               return parent
#          for i in self._avoid_copy_indices:
#              placeholder = self.get_encoded_placeholder(parent)
#              if isinstance(placeholder, Placeholder):
#                  parent.variables[i] = placeholder.get_parent_reference()

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
        
#         global_evolution._override_mutation = True
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