import numpy as np
import custom_types._tools as tools
import contextvars
from warnings import warn
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from typing import Any, Type, Literal, Union
from types import MethodType
from platypus import Solution, Problem, Variator
from platypus.types import Type as PlatypusType
from typing import  Any, Generator, Literal
from copy import deepcopy
from functools import lru_cache, reduce, wraps
from enum import Enum
from ._tools import TypeTuple


class CustomType(PlatypusType):
    """ `
    
    A extended interface of Platypus's `Type` class that stores a `LocalVariator` as an attribute
    
        - A `CustomType` subclass *must* implement a `rand()` method, and can optionally can implement `encode()` and `decode()` methods (see Platypus's `Type` class).
    
    The purpose of a CustomType is to:
        - Streamline the global evolution and mutation of optimization variables across heterogeneous types 
        - Allow instances of the *same* CustomType to evolve and mutate in different ways when those instances represent very different "things"
        - Allow for highly customizable, non-uniform behavior at the global scope (staggered convergence, variable dropouts, etc.)
        - Specify encoding/decoding memoization and deepcopy behavior 
    
    A single GlobalEvolution can be used to evolve all CustomTypes, regardless of the CustomType variables' underlying data structure or LocalVariator. 
    
    A CustomType stores *do_evolution* and *do_mutation* attributes. A CustomType instance can "drop_out" of evolution or mutation by setting these flags to False.
    These flags are automatically set at initialization (depending on the input LocalVariator types)
    
    This class also allows caching / memoization of encoded variable -> decoded variable mapping. 
    A CustomType variable may have an expensive decoding operation; if there is a low probability that an encoded variable will mutate, then temporarily storing the mapping can increase computational efficiency.
    There are two types of memoization that can be implementated
    
    - **LRU Cache**: Stores multiple encoding variable -> decoded variable mappings. Assumes that more recently seen encodings are more likely to appear again.
        - Particularly useful when variables are defined with discrete types and/or when the same encodings appear over the course of multiple generations
        - It is required that the encodings are hashable, and that decoded variables are read-only during evaluations
    - **Single Variable Memoization**: Stores a single encoded variable -> decoded variable mapping between evaluations. 
        - If an encoded variable did not change between evaluations (during a GlobalEvolution's `evolve()`), then no decoding occurs. Instead, the stored decoded variable is deepcopied and returned.
        - More memory efficient than an LRU Cache and does not require decoded variables to be read-only during evaluations
        - It is required that the encodings have a valid `__eq__` method and that decoding does not occur during an evolution or mutation (ie. decoding only occurs *once* before the evaluation)
    
    """    
    
    def __init__(self, 
                 local_variator = None,
                 local_mutator = None,
                 encoding_memoization_type: Literal['cache', 'single'] | None  = None,
                 max_cache_size = 50):
        """
        Must provide at least one of *local_variator* or *local_mutator* 
        
        If both a *local_variator* and *local_mutator* are provided, then a `LocalGAOperator` is automatically created

        Args:
            local_variator (LocalVariator | None, optional): A LocalVariator or None. Defaults to None.
                - Cannot be a LocalMutator and must contain this CustomType subclass in _supported_types (see LocalVariator)
            local_mutator (Local_, optional): A LocalVariator or None_. Defaults to None.
                - Must be a LocalMutator and must contain this CustomType subclass in _supported_types
            encoding_memoization_type (Literal[&#39;cache&#39;, &#39;single&#39;] | None, optional): A memoization technique for encoded variables -> decoded variables. Defaults to None.
                - 'cache': Store encoding / decoding mappings with an LRU cache 
                - 'single': Stores a single encoding / decoding between evaluations
                - (see class doc strings for more details and variable requirements)
            max_cache_size (int, optional): If *encoding_memoization_type* is a cache, specificy the LRU cache size. Defaults to 50.

        """        
        
        # Ensure local_variator and mutator are None, or valid subclass / instance
        if not (local_variator is None or isinstance(local_variator, LocalVariator)) or isinstance(local_variator, LocalMutator):
            raise TypeError(
                f"'local_variator' must be None or a LocalVariator that is not a LocalMutator, got {type(local_variator)!r}"
            )
        if not (local_mutator is None or isinstance(local_mutator, LocalMutator)):
            raise TypeError(
                f"'local_mutator' must be None or LocalMutator, Got {type(local_mutator)!r}"
            )
        
        # check local variators
        self.local_variator = local_variator
        self.do_evolution = None
        self.do_mutation = None
        self._mut = None
        if local_variator is not None and local_mutator is not None:
            self.local_variator = LocalGAOperator(local_variator, local_mutator)
            self.do_evolution = True
            self.do_mutation = True
        elif local_variator is None and local_mutator is None:
            self.do_evolution = False
            self.do_mutation = False
        elif local_variator is None and local_mutator is not None:
            self.local_variator = local_mutator
            self.do_mutation = True
            self.do_evolution = False
        elif isinstance(local_variator, LocalGAOperator):
            self.do_evolution = True
            self.do_mutation = True
        elif isinstance(local_variator, LocalSequentialOperator):
            self.do_mutation = local_variator._contains_mutation
            self.do_evolution = local_variator._contains_crossover 
        
        # print(f"sp_types: {type(self.local_variator)._supported_types}")
        if self.local_variator is not None and not type(self) in self.local_variator._supported_types:
            raise ValueError(f"The CustomType {type(self).__name__} is not a valid type for the input LocalVariator {local_variator!r}. If not a compound/sequential operator, consider registering type")
        
        self._mem_encode = None
        self._mem_decode = None
        self._enc_temp = None
        if encoding_memoization_type == 'cache':
            self.decode = lru_cache(max_cache_size)(self.decode)
        elif encoding_memoization_type == 'single':
            self._mem_encode = self.encode
            self._mem_decode = self.decode
            self.encode = MethodType(_single_memo_encode, self)
            self.decode = MethodType(_single_memo_decode, self)
        
        super().__init__()

    @property
    def can_evolve(self):
        return self._mut
    
    def in_arity_noffspring_limits(self, nparents: int, noffspring: int) -> True:
        """A function for checking if a number of parents Solutions and number of offspring Solutions is compatible with this CustomType's LocalVariator
        
        Useful for debugging"""
        arity_tuple = self.local_variator._supported_arity
        if not (nparents >= arity_tuple[0] and (arity_tuple[1] is None or nparents <=  arity_tuple[1])):
            return False
        noffspring_tuple = self.local_variator._supported_noffspring
        if not (noffspring >= noffspring_tuple[0] and (noffspring_tuple[1] is None or noffspring <=  noffspring_tuple[1])):
            return False
        return True
            
    def execute_variator(self, parent_solutions: list[Solution], offspring_solutions: list[Solution], variable_index: int, copy_indices: list[int], **kwargs):
        """Check the compatibility of inputs into the LocalVariator and calls the LocalVariator. This function may be overridden for any specialized logic or internal attribute updates.
        
        It is not required that a GlobalEvolution call this function (but usually recommended)."""
        noffspring = len(offspring_solutions)
        nparents = len(parent_solutions)
        if len(copy_indices) != noffspring:
            raise ValueError(f"The number of copy_indices ({len(copy_indices)}) is not equal to the number of offspring Solutions ({noffspring})")
        if not self.in_arity_noffspring_limits(nparents, noffspring):
            warn(f"The number of parents solution {nparents} and number of offspring solution {noffspring} is not compatible with this CustomType's LocalVariator {self.local_variator!r}. \n The LocalVariator was not executed")
            return
        
        self.local_variator.evolve(self, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs)
    
    def _can_evolve(self):
        if self.local_variator is None:
            return False
        if self.do_evolution == self.do_mutation:
            return self.do_evolution
        if isinstance(self.local_variator, LocalMutator):
            return self.do_mutation
        if isinstance(self.local_variator, LocalSequentialOperator):
            return (self.local_variator._contains_crossover and self.do_evolution) or (self.local_variator._contains_mutation and self.do_mutation)
        if not isinstance(self.local_variator, LocalGAOperator):
            return self.do_evolution
        else:
            return True
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ('do_mutation', 'do_evolution'):
            if hasattr(self, "do_mutation") and hasattr(self, "do_evolution"):
                self._mut = self._can_evolve()

def _single_memo_encode(self: CustomType, decoded):
    encoded = self._mem_encode(decoded)
    self._enc_temp = (encoded, decoded)
    return encoded
    
def _single_memo_decode(self: CustomType, encoded):
    if not isinstance(self._enc_temp, tuple) or encoded != self._enc_temp[0]:
        self._enc_temp = None
        return self._mem_decode(encoded)
    else:
        decoded = deepcopy(self._enc_temp[1])
        self._enc_temp = None
        return decoded 
    
class StandardCrossover(Enum):
    """
    Keywords for built-in crossover operators recognized by GlobalEvolution.
    
    Some operators uses standard methods on specific elements of an object during a crossover,
    these keywords are also used for those types.

    Use these values to refer to standard crossover methods by name.
    
    """
    SBX = "sbx", (2,2)
    SSX = "ssx", (2,2)
    SPX = "spx", (None, 2)
    PCX = "pcx", (None, 2)
    UNDX = "undx", (None, 2)
    DE = "differential_evolution", (4,1)
    


class StandardMutation(Enum):
    """
    Keywords for built-in mutation operators recognized by the library.
    
    Some operators uses standard methods on specific elements of an object during a mutation,
    these keywords are also used for those types.

    Use these values to refer to standard mutation methods by name.
    """
    UNIFORM = "uniform"
    BITFLIP = "bit_flip"
    POLYNOMIAL = "pm"
    SWAP = "swap"

_current_type = contextvars.ContextVar("current_type") #experimental, for variator attribute overrides

class LocalVariator(metaclass = ABCMeta):
    """`
    
    An abstract class for a `LocalVariator`
    
    LocalVariator subclasses must implement the `evolve()` method. 
    
    Usage
    ---
    A LocalVariator's `evolve()` should only alter offspring solutions at the specified "variable index". 
    It should set an offspring solution's "evaluated" attribute to False if that variable changes.
    So long a LocalVariator follows those guidelines, then it may be defined in any way you see fit.
    
    Parent solutions are provided to direct the crossover, and so a LocalVariator never inadvertently reads its own mutated results. **Parent solutions should never be altered**
    - Unless a LocalVariator is nested in a compound operator, then the offspring solutions will be exact deepcopies of the parents solutions.
    - Some methods will prioritize the values of the offspring, while other methods prioritize the values of the parents. 
    - For example, binary crossover ignores the parent solutions while differential evolution ignores the offspring solution.
    - GlobalEvolution expects the LocalVariator to handle that logic internally
        
    Copy indices are provided to indicate which parent solutions were deepcopied by GlobalEvolution to create the offspring solutions
    - Some methods may use this information to decide which parents to use for the crossover
             
    A LocalVariator may be simple and only define a single method and, optionally, the probability of executing that method.
    Alternatively, a LocalVariator may define multiple methods and their probabilities; 
    this may be for efficiency reasons or when a LocalSequentialOperator cannot properly define the unique sequence of crossover/mutation methods for a type.
    
    (See __init__() for more details)
    
    Attributes
    ---
    
    _supported_types : tuple[Type["CustomType" | "PlatypusType"]]
        The one or more CustomTypes (or Platypus Types) this variator operates on. There is usually only one.
            **This attribute is required to be set**.
    _supported_arity : tuple[int, int | None]
        A pair `(min_arity, max_arity)` describing how many parent Solutions a LocalVariator may accept (use `None` for “no upper-bound”).
        For crossover, this pair is often `(2, None)`. For LocalMutators, it is always `(1,None)`
             **This attribute is required to be set**.
    _supported_noffspring : tuple[int, int | None]
        A pair `(min_offspring, max_offspring)` describing how many offspring solutions may alter at once (use `None` for “no upper-bound”).
        For mutation, it is always `(1,None)`
             **This attribute is required to be set**.        
    
    (LocalCompoundOperator, LocalCompoundMutator and LocalGAOperator set
    'types', 'supported_arity' and 'supported_noffspring' attributes as instance attributes)
    
    Returns
    ---
    list[tuple[Any, bool]]:
        A list of evolved offspring variables, each as `(variable, success_flag)`.
    """    
    _supported_types: TypeTuple[
        Union[
            Type["CustomType"],
            Type["PlatypusType"]
        ]
    ] = TypeTuple()
    _supported_arity: tuple[int, int | None] = None
    _supported_noffspring:  tuple[int, int | None] = None 
    # _standard_methods = None
    # __read_only__ = False
    # __override_names__ = tuple()
    # TODO: Support callables: nparents -> noffspring for  _supported_noffspring
    
    @abstractmethod
    def evolve(self, 
               custom_type: CustomType | PlatypusType,
               parent_solutions: list[Solution], 
               offspring_solutions: list[Solution], 
               variable_index: int,
               copy_indices: list[int | None],
               **kwargs) -> None: 
        """`
        Abstract evolve method. 
        
        Alters *offspring_solutions* in-place at the *variable_index*
        
        Sets the *Solution.evaluted* flag to False if an offspring solution's variable changed
        `
        
        Args
        ----
        **custom_type**: The CustomType or PlatypusType object the variables are derived from
            - If it is a CustomType, may contain overriding parameters
        
        **parent_solutions**: Read-only Solutions objects
        
        **offspring_solutions** (list[object]): Deepcopied *parent_solutions*: the Solutions to update.
        
        **variable_index** (int | list[int]): The index the *custom_type* variable in the `Solution` objects
            - The LocalVariator should only update `Solution.variables[variable_idx]`
            
        **copy_indices** (list[int | None]): A list of *parent_solution* indices (the same length as *offspring_solutions*)
            
            - `offspring_solution[idx]` is a copy of `parent_solutions[copy_indices[idx]]`
            
            - Each index is valid index of the *parents_solution* list or None. May contain duplicate indices if a parent solution was deepcopied more than once

            - If an element is None, it indicates that no parent Solutions corresponds to an offspring Solution
                (parent solution may have been removed or altered)
        
        **kwargs**: any additional keyword arguments passed in from a `GlobalEvolution` object

            Extra arguments may include: 
            
        - other method(s) or parameters for generic types, 
        - other *custom_type* variable indices (to calculate aggregate values),
        - etc.
        
        Raises:
            NotImplementedError
        """
        raise NotImplementedError()
    
    @staticmethod
    def get_solution_variables(solutions: Solution | Iterable[Solution], index) -> Any | list[Any]:
        """Get the variable at an index for one or more solutions"""
        if isinstance(solutions, Solution):
            return solutions.variables[index]
        return [s.variables[index] for s in solutions]
    
    @staticmethod
    def get_no_copy_variables(parent_solutions: Iterable[Solution], variable_index, copy_index: int | None) -> list:
        """Get all variables of parent solutions at 'variable_index', excluding the solution at 'copy_index'. 
        If copy_index is None or out of range, returns all variables of parent_solutions"""
        
        if copy_index is None:
            return LocalVariator.get_solution_variables(parent_solutions, variable_index)
        return [par.variables[variable_index] for i, par in enumerate(parent_solutions) if i != copy_index]
    
    @staticmethod
    def group_by_copy(copy_indices: list[int, None]) -> dict[int,list[int]]:
        """Group offspring positions by their copy identifier. (Useful when some parents may have be copied > 1 time)

        Args:
            copy_indices (list[int, None]): For each offspring (by position), the index of its 'copy' parent or `None` if it has no associated copy.

        Returns:
            dict[int,list[int]]: 
                A mapping from a copy index (ie. parent solution index) to a list of offspring positions that share it. Offspring entries whose `copy_indices[i]` is `None` are grouped under the key `-1`.
        """        
        unique_copies = {} 
        for i, copy_idx in enumerate(copy_indices):
            if copy_idx is None:
                unique_copies.setdefault(-1, []).append(i)
            else:
                unique_copies.setdefault(copy_idx, []).append(i)
        return unique_copies
    
    @classmethod
    def register_type(cls, valid_type: type):
        """Temporarily register a CustomType or Platypus Type subclass to be valid for this LocalVariator class. 
        
        (Not applicable for any compound/sequential operators)
        """
        if not issubclass(valid_type, (CustomType, PlatypusType)):
            raise TypeError(f"The input type must be a CustomType or Platypus Type subclass, got {type(valid_type)}")
        prev_types = cls._supported_types
        if prev_types is None:
            cls._supported_types = TypeTuple(valid_type)
        elif isinstance(prev_types , type):
            cls._supported_types = TypeTuple(prev_types, valid_type)
        elif isinstance(prev_types, TypeTuple):
            cls._supported_types = prev_types + (valid_type, )
        else:
            cls._supported_types = TypeTuple(*prev_types, valid_type)
            
    
    def __init_subclass__(cls, **kwargs):
        types = getattr(cls, "_supported_types", None)
        if types is not None and not isinstance(types, TypeTuple):
            cls._supported_types = TypeTuple(types) if isinstance(types, type) else TypeTuple(*types)
        for t in cls._supported_types:
            if not isinstance(t, type):
                raise TypeError(f"The _supported_types attribute of the LocalVariator subclass '{cls.__name__}' contains an entry that is not a type. Got an entry of type {type(t)}")
        
        # cls.__attr_override__()
        # super().__init_subclass__(cls, **kwargs)
        
    def __repr__(self):
        ar = str(self._supported_arity) if self._supported_arity is None else repr(self._supported_arity)
        of = str(self._supported_noffspring) if self._supported_noffspring is None else repr(self._supported_noffspring)
        return f"{self.__class__.__name__}(types = {str(self._supported_types)}, arity = {ar}, noffspring = {of})"
    
    """
    EXPERIMENTAL: 
    - contextvars for automatic override of probability attributes occurs + concurrency safe
    - allows CustomTypes to share LocalVariators, but update probability attributes separately
    """
    #@classmethod
    # def __attr_override__(cls):
    #     orig_evolve = cls.evolve
    #     @wraps(orig_evolve)
    #     def evolve_wrapper(self, custom_type, *args, **kwargs):
    #         type_token = _current_type.set(custom_type)
    #         try:
    #             return orig_evolve(self, custom_type, *args, **kwargs)
    #         finally:
    #             _current_type.reset(type_token)
    #     cls.evolve = evolve_wrapper
    
    # def __getattribute__(self, name):
    #     if name in object.__getattribute__(self, "__override_names__"):
    #         t = _current_type.get(None)
    #         if t is not None and hasattr(t, name):
    #             return getattr(t, name)
    #     return object.__getattribute__(self, name)

class LocalMutator(LocalVariator, metaclass = ABCMeta):
    """
    A subclass LocalVariator where there is exactly one offspring variable. These variators do not require a reference 'parent' Solution.
    
    The `evolve()` method can accept any number of offspring Solutions, and will mutate each one separately.
    
    Subclasses must implement a `mutate()` method. Unlike a LocalVariator, `mutate()` (and `evolve()`) cannot be class methods.
        `evolve()` should generally not be overriden.
    
    All subclass implementations of `mutute()` should include `**kwargs` in their signature so that any unexpected/irrelevant keyword arguments are safely ignored

    (see `LocalVariator)
    """
    _supported_arity =  (1,None)
    _supported_noffspring = (1,None)
    
    def evolve(
        self, 
        custom_type: CustomType | PlatypusType,
        parent_solutions: list[Solution],
        offspring_solutions: Solution | list[Solution], 
        variable_index: int | list[int],
        copy_indices,
        **kwargs) -> None:
        """ 
        Usually called by a compound operator or GAOperator.
        
        The *parents* and *copy_indices* parameters are ignored. "**kwargs" is passed to `mutate()`
        
        (See `LocalVariator` for details on parameters and return type)
        """
        if isinstance(custom_type, CustomType) and not custom_type.do_mutation:
            return 
        if isinstance(offspring_solutions, Solution):
            self.mutate(custom_type, offspring_solutions, variable_index, **kwargs)
            return
        map(lambda offspring: self.mutate(custom_type, offspring, variable_index, **kwargs), offspring_solutions)
            
    @abstractmethod
    def mutate(
        self,
        custom_type: CustomType | PlatypusType,
        offspring_solution: Solution, 
        variable_index: int | list[int],
        **kwargs) -> None:
        """`
        Abstract method that mutates one Solution in-place
        
        Args:
        ---
        **custom_type**: A Platypus or CustomType object
        
        **offspring_solution**: A deepcopied parent Solution
        
        **variable_index**: An index of *offspring_solution.variables*
            
        **kwargs**: any additional keyword arguments passed in from a `GlobalMutation` or CustomType object
            Extra arguments may include: 
            
        - override probabilities, 
        - extra parameters controlling the mutation methods of generic types,
        - etc.

        Raises:
            NotImplementedError
        """
        
        raise not NotImplementedError  
    
    def __setattr__(self, name, value):
        if name in ("_supported_arity", "_supported_noffspring"):
            raise AttributeError(f"Cannot override class attribute {name!r} for LocalMutators")
        super().__setattr__(name, value)
    
    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)
    
    # @classmethod
    # def __attr_override__(cls):
    #     orig_mutate= cls.mutate
    #     @wraps(orig_mutate)
    #     def mutate_wrapper(self, custom_type, *args, **kwargs):
    #         type_token = _current_type.set(custom_type)
    #         try:
    #             return orig_mutate(self, custom_type, *args, **kwargs)
    #         finally:
    #             _current_type.reset(type_token)
    #     cls.mutate = mutate_wrapper
        

class GlobalEvolution(Variator):
    """
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
        super(GlobalEvolution, self).__init__(arity)
        self.noffspring = noffspring
        self._ignore_generics = ignore_generics
        self._indices_by_type = None

    def get_parent_deepcopies(
        self,
        parents: list[Solution], 
        copy_indices: int | list[int] | Literal['all'] = 'all') -> list[Solution]:      
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
        if isinstance(parent_indices, int):
            out =[deepcopy(parents[parent_indices])]
        else:
            if parent_indices == 'all':
                parent_indices = range(len(parents))
            out = [deepcopy(parents[i]) for i in copy_indices]
        
        return out
    
    def generate_types_to_evolve(self, problem: Problem) -> Generator[tuple[CustomType | PlatypusType, int]]:
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
        for i, var_type in enumerate(problem_types):
            if (isinstance(var_type, CustomType) and var_type.can_evolve) or (not isinstance(var_type, CustomType) and self._ignore_generics):
                yield var_type, i
    
    def generate_type_groups(self, problem: Problem) -> Generator[tuple[list[CustomType | PlatypusType], tuple[int]]]:
        self._store_problem_types(problem)
        problem_types = problem.types
        for type, indices in self._indices_by_type:
            yield [problem_types[i] for i in indices], indices 
    
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
        
                
    # @staticmethod
    # def get_encoded_placeholder(solution: Solution, variable_index: int):
    #     """ If the `solution.variables[variable_index]` is a PlaceholderPair, return the Placeholder solutions.variables[] encoded data.
        
    #     Otherwise, return `solution.variables[variable_index]
        
    #     PlaceholderPairs carry a Placeholder for both the decoded and encoded variable (these are frequently the same). 
    #         This method returns the encode Placeholder.
        
    #     Args:
    #         sol (Solution):
    #         variable_index (variable_index): An index in solution.variables
        
    #     """        
    #     var = solution.variables[variable_index]
    #     if isinstance(var, Placeholder):
    #         return var.get_parent_reference()
    #     return var

    # def replace_all_parent_placeholders(self, parents: list[Solution]):
    #     """
    #     Return all parents to their original state. 
        
    #     Check all variables that could have a Placeholder
    #     If a variable does have Placeholder present, remove the (encoded) variable from that Placeholder.
        
    #     (If applicable, should be called after deepcopies are created. This input should be the original Solutions, not the deepcopy Solutions)

    #     """   
    #     if not self._avoid_copy_indices:
    #         return 
    #     for i in self._avoid_copy_indices:
    #         for par in parents:
    #             placeholder = self.get_encoded_placeholder(par, i)
    #             if isinstance(placeholder, Placeholder):
    #                 par.variables[i] = placeholder.get_parent_reference()
                 

class LocalGAOperator(LocalVariator):
    """Combine a `LocalVariator` with a `LocalMutator`. 
    
    A LocalMutator's `mutate()` is called for each output offspring the LocalVariators `evolve()` method.
    """
    
    def __init__(
        self, 
        local_variator: LocalVariator, 
        local_mutator: LocalMutator):
        """
        Args:
            local_variator (LocalVariator): A LocalVariator, cannot be a LocalMutator 
            local_mutator (LocalMutator): A LocalMutator.
            
        Raises:
            TypeError: If the LocalVariator or LocalMutator do not share a type in _supported_types
        """      
        
        if isinstance(local_variator, LocalMutator) or not isinstance(local_variator, LocalVariator):
            raise TypeError(f"The input 'local_variator' must be a LocalVariator that is not a LocalMutator, got {type(local_variator)})")
        if not isinstance(local_mutator, LocalMutator):
            raise TypeError(f"The input 'local_mutator' must be a LocalMutator, got {type(local_mutator)})")
        
        common_types =  local_variator._supported_types & local_mutator._supported_types
        if not common_types:
            v = local_variator.__name__
            m = local_mutator.__name__
            raise TypeError(
                f"The input local_variator '{v}' and input local_variator '{m}' do not have compatible _supported_types. "
                "Ensure that these variators have at least one shared type"
                )

        self.local_variator = local_variator
        self.local_mutator = local_mutator
        self._supported_arity = local_variator._supported_arity
        self._supported_noffspring = local_variator._supported_noffspring
        self._supported_types = TypeTuple(*common_types)
        
    def evolve(self, custom_type: CustomType | PlatypusType, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        if not type(custom_type) in self._supported_types:
            return 
        
        if not isinstance(custom_type, CustomType) or custom_type.do_evolution:
            self.local_variator.evolve(custom_type, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs)
         
        self.local_mutator.evolve(custom_type, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs)

class LocalCompoundMutator(LocalMutator):
    """
    Sequentially compound any number of LocalMutators
    
    Other LocalCompoundMutators may be part of the sequence
    
    All LocalMutator's must share at least 1 common type in _supported_types 
        (see LocalVariator)
    """
    def __init__(
        self, 
        local_mutators: list[LocalMutator], 
        randomize_start: bool = False):
        """
        Args:
            local_mutators (list[LocalMutator]): A list of LocalMutator objects. A minimum of 2.
            randomize_start (bool, optional): If True, choose a random mutator as the first mutator (and wrap around). Otherwise, always start with the first mutator in the list. Defaults to False.
        """        
        # Input validation
        if not isinstance(local_mutators, list) or len(local_mutators) < 2:
            raise TypeError("There must be at least 2 LocalMutators provided in a list")   
        for i, mutator in enumerate(local_mutators):
            if not isinstance(mutator, LocalMutator):
                raise TypeError(f"The element at index {i} is not a LocalMutator, got {type(mutator)}")
            
        common_types =  reduce(
            lambda a, b: a & b, 
            (v._supported_types for v in local_mutators)
        )
        if not common_types:
            raise TypeError("The inputted LocalMutators do not share at least one supported type. Ensure the variators have the _supported_types attribute set and the methods are compatible")
        
        self._supported_types = TypeTuple(*common_types)
        self.mutators = local_mutators
        self.num_mutators = len(local_mutators)
        self.randomize_start = randomize_start
    
    def mutate(self, custom_type, offspring_solution, variable_index, **kwargs):
        if not custom_type.do_mutation or not type(custom_type) in self._supported_types:
            return
        start_idx = 0 if not self.randomize_start else np.random.randint(self.num_mutators)
        if start_idx == 0:
            for mutator in self.mutators:
                mutator.mutate(custom_type, offspring_solution, variable_index, **kwargs)
        else:
            for i in range(self.num_mutators):
                idx = (start_idx + i) % self.num_mutators
                mutator = self.mutators[idx]
                mutator.mutate(custom_type, offspring_solution, variable_index, **kwargs)
    
    def __repr__(self):
        mutators = ", ".join(type(var).__name__ for var in self.mutators)
        return f"{self.__class__.__name__}(mutators = ({mutators}), types = {str(self._supported_types)})"     


MutationSelection = Literal['previous', 'all', 'rand']
OffspringSelection = Literal['previous', 'swap', 'rand']
class LocalSequentialOperator(LocalVariator):
    """ 
    Sequentially compound any number of LocalVariators 
    
    This operator takes deep–copied offspring Solutions and reuses them across multiple LocalVariators, only deepcopying the relevant variable between stages.
    
    A LocalSequentialOperator has less restrictions than a CompoundOperator in regard to compatible arities.

    Requirements
    ---
    - All variators in the sequence must share at least one CustomType (or Platypus Type) in _supported_types
    - All variators' minimum _supported_arity `≤` the number of parent Solutions passed to `evolve()`
    - All variators' minimum _supported_noffspring `≤` the number of offspring Solutions passed to `evolve()`
    - LocalGAOperators, LocalCompoundOperators and LocalSequentialOperators *cannot* be part of the sequence (any equivalent nesting can be expressed with a single LocalCompoundOperator)
        - LocalMutators and LocalCompoundMutators can be used in the sequence

    For details on class attributes and `evolve()` parameters, see `LocalVariator`
    
    Operator Sequencing
    ---
    
    - Initial Inputs:
        - The *offspring_solutions* are deepcopies of one or more *parents_solutions* (handled by GlobalEvolution)
    
    - Variator A:  
        - `p₁ = min(len(parent_solutions), A._supported_arity[1])` number of parents are chosen  
        - `o₁ <= A._supported_noffspring[1])` number of offspring are are chosen 
            - which offspring are chosen depends on parameterization, see __init__()
        - `A` alters the `o₁` offspring in-place with the `p₁` parents . 
    
    - Before Variator B;
        - Update *parent_solutions* with the previous `o₁` offspring solutions. 
        Remove the corresponding solutions from *parent_solutions* (using *copy_indices*)
    
    - Variator B:
        - `p₂ = min(len(parent_solutions), B._supported_arity[1])`  
        - `o₂ <= B._supported_noffspring[1])`  
        -  For any `o₂` offspring solutions that are also passed as `p₂` parent solutions:
                - Shallow-copy the Solution, deepcopy the relevant variable w/in Solution.variables
                - (LocalMutators do not read parent solutions and skip this step)
        - `B` alters the `o₂` offspring in-place with the `p₂` parents 
                - If copying occured, replace the variable of the original offspring the its shallow-copy variable
        
     - (repeat previous two steps)
    
    In summary, each LocalVariator:
    - sees only as many parents and offspring as it supports,  
    - sees all updates to parent solutions,  
    - never reads its own mutated results.
     """
     
    def __init__(
        self,
        local_variators: list[LocalVariator], 
        offspring_selection: OffspringSelection | list[OffspringSelection] = 'previous',
        mutation_selection: MutationSelection = 'previous',
        randomize_start = False):  
        """`
        Args:
        
            **local_variators** (list[LocalVariator]): A list of LocalVariators to compound (minimum of 2).
                - Cannot contain LocalGAOperators or LocalCompoundOperators
                
            **offspring_selection** (OffspringSelection | list[OffspringSelection], optional): {'previous', 'swap', 'rand'} or a sequence thereof. Defaults to 'previous'.
                
                Strategy for choosing which Solutions are the mutable "offspring" in each sub-operator (excluding LocalMutators)
                - 'previous' : Prioritize the offspring mutated by the previous variator as the mutable offspring of the next variator
                - 'rand'     : Choose random offspring Solutions (without replacement)
                - 'swap'     : Prioritize any unaltered offspring as the mutable offspring of the next variator
                
                If a string is provided, that selection strategy will be applied at all stages.
                
                If a list is inputted, it must be of length `len(localvariators)`. The strategy at index `i` will be applied before variator `i` is called. 
                    If LocalMutators exist in the sequence, their selection strategies should also be included (see *mutation_selection*).
                    
            **mutation_selection** (MutationSelection, optional): {'previous', 'all', 'rand'}. Defaults to 'previous'.
            
                Strategy for choosing which offspring Solutions should be mutated when a LocalVariator is a LocalMutator
                - 'previous' : Mutate all Solutions that were passed as offspring to the previous operator
                - 'rand'     : Mutate one random offspring Solution 
                - 'all'      : Mutate all Solutions that were passed as offspring to the LocalCompoundOperator. 
                
                If *offspring_selection* is a list, then mutation_selection* is ignored. The mutation strategies should be included in *offspring_selection*.
                    
            **randomize_start** (bool, optional): If True, choose a random variator as the first variator (and wrap around). Otherwise, always start with the first variator in the list. Defaults to False.
        """        
        CROSSOVER_STRATEGIES = {'previous', 'swap', 'rand'}  
        MUTATION_STRATEGIES = {'previous', 'all', 'rand'}  
          
        if not isinstance(local_variators, list) or len(local_variators) < 2:
            raise TypeError("There must be at least 2 LocalVariators provided in a list")   

        # Check input types / supported arities
        min_arity = 1
        min_noffspring = 1
        self._contains_crossover = False
        self._contains_mutation = False
        nvariators = 0
        for i, variator in enumerate(local_variators):
            
            if not isinstance(variator, LocalVariator):
                 raise TypeError(f"The LocalVariator at position {i} is not a LocalVariator, got {type(variator)}.")
            if isinstance(variator, (LocalGAOperator, LocalSequentialOperator)):
                raise TypeError(f"The LocalVariator at position {i} is a LocalSequentialOperator or a LocalGAOperator; nested compound operators are not supported")
            if not isinstance(variator, LocalMutator):
                self._contains_crossover = True
                nvariators += 1
            else:
                self._contains_mutation = True
                 
            curr_arity =  variator._supported_arity
            curr_noffspring = variator._supported_noffspring
            
            if not (isinstance(curr_arity, tuple) and isinstance(curr_noffspring, tuple)):
                raise TypeError(f"The LocalVariator at position {i} does not have _supported_arity or __supported_offspring__ set. Ensure these attributes are set and are tuples. ")
            if curr_arity[0] is None or curr_noffspring[0] is None:
                raise TypeError(f"The LocalVariator at position {i} minimum _supported_arity or minimum __supported_offspring__ is None. Only upper bounds can be None")
            
            if curr_arity[0] > min_arity:
                min_arity = curr_arity[0]
            if curr_noffspring[0] > min_noffspring:
                min_noffspring = curr_noffspring[0]
           
        common_types = reduce(
            lambda a, b: a & b,
            (v._supported_types for v in local_variators)
            # local_variators[0]._supported_types
        )
        if not common_types:
            raise TypeError("The inputted LocalVariators do not share at least one supported type. Ensure the variators have the _supported_types attribute set and the methods are compatible")
        
        self._supported_types = TypeTuple(*common_types)
        self._supported_arity  = (min_arity, None)
        self._supported_noffspring  = (min_noffspring, None)
        self.variators = local_variators
        self.randomize_start = randomize_start
        self.offspring_selection = offspring_selection
        self.mutation_selection = None
        self.contains_swap = 0
        
        # Check selection strategies
        if isinstance(offspring_selection, list):
            if len(offspring_selection) != len(local_variators):
                raise TypeError("If 'offspring_selection is a list, then it must be the same length as 'local_variators'")
            for i, strategy in enumerate(offspring_selection):
                if not isinstance(local_variators[i], LocalMutator) and not strategy in CROSSOVER_STRATEGIES:
                    raise ValueError(f"'{strategy} is not a valid 'offspring_selection' for a LocalVariator ")
                if isinstance(local_variators[i], LocalMutator) and not strategy in MUTATION_STRATEGIES:
                    raise ValueError(f"'{strategy} is not a valid 'mutation_selection' for a LocalMutator")
                self.contains_swap += strategy == 'swap'
        else:
            if not offspring_selection in CROSSOVER_STRATEGIES:
                raise ValueError(f"'{offspring_selection} is not a valid 'offspring_selection' for a LocalVariator ")
            if not mutation_selection in MUTATION_STRATEGIES:
                raise ValueError(f"'{mutation_selection} is not a valid 'mutation_selection' for a LocalMutator ")
            self.mutation_selection = mutation_selection
            self.contains_swap = nvariators
        
    
    def __repr__(self):
        variators = ", ".join(type(var).__name__ for var in self.variators)
        return f"{self.__class__.__name__}(variators = ({variators}), types = {str(self._supported_types)}, min_arity = {self._supported_arity[0]}, min_noffspring = {self._supported_noffspring[0]}"
                
     
    def evolve(self, custom_type: CustomType | PlatypusType, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs):
        if not type(custom_type) in self._supported_types:
            return 
        do_mutation = self._contains_mutation
        do_evolution = self._contains_crossover
        if isinstance(custom_type, CustomType):
            do_mutation &= custom_type.do_mutation 
            do_evolution &= custom_type.do_evolution
        if not (do_mutation or do_evolution):
            return
        
        nparents = len(parent_solutions)
        noffspring = len(offspring_solutions)
        nvariators = len(self.variators)
        start_idx = 0 if not self.randomize_start else np.random.randint(nvariators)
        
        if not do_evolution:
            self._mutation_only(
                custom_type, parent_solutions, offspring_solutions, variable_index, copy_indices, 
                nvariators, noffspring, start_idx, **kwargs)
            return
        
        # Arity and copy_indices checks
        if nparents < self._supported_arity[0]:
            raise ValueError(f"{nparents} parent Solutions were inputted, at least {self._supported_arity[0]} are required")
        if noffspring < self._supported_noffspring[0]:
            raise ValueError(f"{noffspring} offspring Solutions were inputted, at least {self._supported_noffspring[0]} are required")
        if len(copy_indices) != noffspring:
            raise ValueError(f"{copy_indices} must be same length as the number of offspring. {len(copy_indices)} indices were provided (vs. {noffspring} offspring)") 

        # First variator, no alterations have occured. Loop until valid variator is found
        altered_offspring = None
        multi_strategy = isinstance(self.offspring_selection, list)
        previous_swap = False
        idx = start_idx
        while True:
            variator = self.variators[idx]
            if do_mutation and isinstance(variator, LocalMutator):
                strategy1 = self.mutation_selection if not multi_strategy else self.offspring_selection[idx]
                altered_offspring = self._first_mutation(
                    variator, custom_type, 
                    parent_solutions, offspring_solutions, 
                    variable_index, copy_indices,
                    noffspring, strategy1, **kwargs
                )
                idx = (idx + 1) % nvariators
                break
            elif not isinstance(variator, LocalMutator):
                strategy1 = self.offspring_selection if not multi_strategy else self.offspring_selection[idx]
                altered_offspring = self._first_variation(
                    variator, custom_type, 
                    parent_solutions, offspring_solutions, 
                    variable_index, copy_indices,
                    nparents, noffspring, 
                    strategy1, **kwargs
                )
                previous_swap = strategy1 == 'swap'
                idx = (idx + 1) % nvariators
                break
            else:
                idx = (idx + 1) % nvariators
                if idx == start_idx:
                    break
        
        if idx == start_idx:
            print("here return")
            return
        
        # rest of variators
        swaps_left = self.contains_swap - (previous_swap)
        self._multi_strategy(
            custom_type, parent_solutions, offspring_solutions, 
            variable_index, copy_indices,
            altered_offspring, 
            nparents, noffspring, 
            start_idx, idx, nvariators, 
            swaps_left, do_mutation, **kwargs
        )
        
    
    def _multi_strategy(
        self, 
        custom_type, parent_solutions: list[Solution], offspring_solutions: list[Solution], 
        variable_index: int, copy_indices: list, 
        altered_offspring: set, 
        nparents, noffspring, 
        start_idx, curr_idx, nvariators, 
        swaps_left, do_mutation, **kwargs):
        
        # Keep track of altered/unaltered offspring + parents that were updated
        num_altered = len(altered_offspring)
        unaltered_offspring = set() if num_altered == noffspring else set(range(noffspring)).difference(altered_offspring)
        previously_altered = altered_offspring.copy()
        
        
        orig_offspring_indices = [i for i in altered_offspring]
        original_parents = set(range(nparents))
        for i in orig_offspring_indices:
            copied_from = copy_indices[i]
            if copied_from is not None:
                original_parents.discard(copied_from)
        print(f"START_IDX: {start_idx}, First altered: {previously_altered}, Original Parents {original_parents}")
        
        multi_strategy = isinstance(self.offspring_selection, list)
        nvariators = len(self.variators)
        while curr_idx != start_idx:
            curr_variator = self.variators[curr_idx]
            
            # Mutation
            if isinstance(curr_variator, LocalMutator) and do_mutation: 
                curr_strategy = self.mutation_selection or self.offspring_selection[curr_idx]
                print(f"CURR IDX: {curr_idx}, MUT_STRATEGY: {curr_strategy}")
                print(f'unaltered_before: {unaltered_offspring}')
                
                if curr_strategy == 'all':
                    curr_variator.evolve(custom_type, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs)
                    if num_altered != noffspring:
                        altered_offspring = set(range(noffspring))
                        num_altered = noffspring
                        orig_offspring_indices = list(range(noffspring))
                        unaltered_offspring.clear()
                    previously_altered = set(range(noffspring))
                    
                elif curr_strategy == 'previous':
                    prev_offspring_solutions = [offspring_solutions[i] for i in previously_altered]
                    curr_variator.evolve(custom_type, parent_solutions, prev_offspring_solutions, variable_index, [None for _ in range(len(previously_altered))], **kwargs)
                    
                else:
                    rand_offspring_idx = np.random.randint(noffspring)
                    curr_variator.mutate(custom_type, offspring_solutions[rand_offspring_idx], variable_index, **kwargs)
                    if num_altered < noffspring and not rand_offspring_idx in altered_offspring:
                        num_altered += 1
                        altered_offspring.add(rand_offspring_idx)
                        orig_offspring_indices.append(rand_offspring_idx)
                        if copy_indices[rand_offspring_idx] is not None:
                            original_parents.discard(copy_indices[rand_offspring_idx])
                        unaltered_offspring.discard(rand_offspring_idx)
                    previously_altered = {rand_offspring_idx}

                print(f"unaltered after: {unaltered_offspring}\n")
            
            # Crossover       
            elif not isinstance(curr_variator, LocalMutator): 
                curr_strategy = self.offspring_selection if not multi_strategy else self.offspring_selection[curr_idx]
                new_parent_choices = None 
                original_parent_choices = None
                altered_offspring_choices = None 
                unaltered_offspring_choices = None
                min_arity, max_arity, min_noffspring, max_noffspring = self._get_arity_limits(curr_variator, nparents, noffspring)
                
                # Select parents from 'altered offspring' and uncopied/unused 'parent_solutions', 
                # Select offspring from both altered and unaltered offspring
                print(f"CURR IDX: {curr_idx}, STRATEGY: {curr_strategy}, VARIATOR {curr_variator.__class__.__name__}")
                print(f'unaltered_before: {unaltered_offspring}, original_parents_before {original_parents}')
                if curr_strategy == 'rand':
                    new_parent_choices, original_parent_choices, altered_offspring_choices, unaltered_offspring_choices = tools._rand_selection(
                        orig_offspring_indices, original_parents, unaltered_offspring, 
                        nparents, noffspring, swaps_left, 
                        min_arity, max_arity, min_noffspring, max_noffspring)
                    
                elif curr_strategy == 'swap':
                    new_parent_choices, original_parent_choices, altered_offspring_choices, unaltered_offspring_choices = tools._swap_selection(
                        orig_offspring_indices, original_parents, unaltered_offspring, previously_altered,
                        nparents, noffspring, swaps_left, 
                        min_arity, max_arity, min_noffspring, max_noffspring)
                    swaps_left -= 1

                else:
                    new_parent_choices, original_parent_choices, altered_offspring_choices, unaltered_offspring_choices = tools._previous_selection(
                        orig_offspring_indices, original_parents, unaltered_offspring, previously_altered,
                        nparents, noffspring, swaps_left, 
                        min_arity, max_arity, min_noffspring, max_noffspring
                    )
                    
                print(f"new_parent_choices: {new_parent_choices}, original_parent_choices {original_parent_choices}")
                print(f"altered_offspring_choices: {altered_offspring_choices}, unaltered_offspring_choices {unaltered_offspring_choices}")
                
                n_altered_choices = len(altered_offspring_choices)
                if len(altered_offspring_choices) > 0:
                    assert n_altered_choices == len(np.unique(altered_offspring_choices))
                if len(unaltered_offspring_choices) > 0:
                    assert len(unaltered_offspring_choices) == len(np.unique(unaltered_offspring_choices))
                                                              
                                                              
                # Create new list of parent/offspring solutions, get new copy_indices
                new_offspring, new_copy_indices, post_replacements = tools._copy_indices_and_deepcopy(
                    offspring_solutions, variable_index, new_parent_choices, altered_offspring_choices
                )
                new_parents = [offspring_solutions[i] for i in new_parent_choices]
                # track = None
                # track_vars = None
                # if len(new_parent_choices) > 0:
                #     track = [i for i,p in enumerate(new_parent_choices) if any(a == p for a in altered_offspring_choices)]
                #     track_vars = [deepcopy(offspring_solutions[new_parent_choices[t]].variables[0]) for t in track]
                
                tools._unaltered_overlap(
                    original_parents, 
                    copy_indices, 
                    unaltered_offspring_choices,
                    original_parent_choices, 
                    new_copy_indices, 
                    n_new_parents=len(new_parents)
                )
                if original_parent_choices:
                    new_parents.extend(parent_solutions[i] for i in original_parent_choices)
                new_offspring.extend(offspring_solutions[i] for i in unaltered_offspring_choices)
                
                # n_altered_choices = len(altered_offspring_choices)
                # if n_altered_choices > 0:
                #     print(f"ALTERED BEFORE: {[offspring_solutions[i].variables[variable_index] for i in altered_offspring_choices]}")
                
                print(f"NUM PARENTS: {len(new_parents)}")
                print(f"NUM OFFSPRING: {len(new_offspring)}")
                print(f"COPY_INDICES: {new_copy_indices}, (original {copy_indices})")
                print(f"POST_REPLACEMENTS: {post_replacements}")
                assert len(new_offspring) == len(new_copy_indices), f"new_copy_indices: {new_copy_indices}, post_replacements {post_replacements}"
                assert all(idx is None or idx < len(new_parents) for idx in new_copy_indices)
                 
                # Pass new offspring into variator's evolve, do variable replacements + other updates
                curr_variator.evolve(custom_type, new_parents, new_offspring, variable_index, new_copy_indices, **kwargs)
                # if n_altered_choices > 0:
                #     str_after = ""
                #     for k in range(n_altered_choices):
                #         str_after += f"{[new_offspring[k].variables[variable_index]]}, "
                #     print(f"OFFSPRING_AFTER_A: {str_after}\n")
                
                # if track is not None:
                #     for t, var in zip(track, track_vars):
                #         assert np.all(var == new_parents[t].variables[variable_index])
        
                tools._variable_replacement(offspring_solutions, new_offspring, post_replacements, variable_index)
                # if n_altered_choices > 0:
                #     print(f"OFFSPRING_AFTER_2: {[offspring_solutions[i].variables[variable_index] for i in altered_offspring_choices]}")
 
                if unaltered_offspring_choices:
                    unaltered_offspring.difference_update(unaltered_offspring_choices)
                    orig_offspring_indices.extend(unaltered_offspring_choices)
                    altered_offspring.update(unaltered_offspring_choices)
                    num_altered += len(unaltered_offspring_choices)
                previously_altered = set(altered_offspring_choices + unaltered_offspring_choices)
                print(f'unaltered_after: {unaltered_offspring}, original_parents_after {original_parents}\n')
            curr_idx = (curr_idx + 1) % nvariators
            
  
    def _get_arity_limits(self, variator: LocalVariator, nparents, noffspring):
        """returns: (min_arity, max_arity, min_noffspring, max_noffspring)"""
        max_arity = min(variator._supported_arity[1] or nparents, nparents)
        min_arity = variator._supported_arity[0]
        max_noffspring = min(variator._supported_noffspring[1] or noffspring, noffspring)
        min_noffspring = self._supported_noffspring[0]
        return min_arity, max_arity, min_noffspring, max_noffspring
    
    def _first_variation(
        self, variator: LocalVariator, 
        custom_type, parent_solutions: list, offspring_solutions: list, 
        variable_index: int, copy_indices: list, 
        nparents, noffspring, strategy, **kwargs):
        """ Assumes variator is not mutator, no alteration of offspring has occured, 
        nparents >= min_arity, noffspring >= min_noffspring, len(copy_indices) == len(offspring_solutions)
        
        Returns: set of offspring indices that were passed into the variator
        """
        min_arity, max_arity, min_noffspring, max_noffspring = self._get_arity_limits(variator, nparents, noffspring)
        arity_difference = nparents - noffspring
        
        # Choose output noffspring and nparents (close to original arity difference)
        out_nparents = None
        out_noffspring = None
        if max_noffspring == min_noffspring:
            out_noffspring = min_noffspring
        elif self.contains_swap: # keep offspring avaliable for later swap
            out_noffspring = max(min_noffspring, max_noffspring + arity_difference) if arity_difference < 0 else max_noffspring
            out_noffspring = max(min_noffspring, min(max_noffspring - (self.contains_swap - int(strategy == 'swap')), out_noffspring))
        else:
            out_noffspring = max_noffspring
            
        if min_arity == max_arity or arity_difference > 0:
            out_nparents = max_arity
        elif arity_difference < 0:
            out_nparents = min_arity if out_noffspring - 1 <= min_arity else max(min_arity, min(max_arity, out_noffspring + arity_difference))
        else: 
            out_nparents = max(min_arity, min(max_arity, out_noffspring + arity_difference))
        
        print(f"FIRST: OUT_nparents {out_nparents}, OUT_noffspring: {out_noffspring}")
        
        # Trivial nparents vs. noffspring
        if out_nparents == nparents and out_noffspring == noffspring:
            variator.evolve(custom_type, parent_solutions.copy(), offspring_solutions.copy(), variable_index, copy_indices, **kwargs)
            return set(range(noffspring))
        
        # out_noffspring < noffspring
        out_offspring_indices = []
        if out_nparents == nparents:
            seen_parents = set()
            num_added = 0
            offspring_left = []
            new_copy_indices = []
            for i, parent_idx in enumerate(copy_indices):
                if parent_idx is None or parent_idx in seen_parents:
                    offspring_left.append(i)
                    continue
                
                out_offspring_indices.append(i)
                new_copy_indices.append(parent_idx)
                seen_parents.add(parent_idx)
                num_added += 1
                if num_added == out_noffspring:
                    break
                
            if num_added < out_noffspring: 
                if not num_added: # all unlabeled
                    out_offspring_indices = offspring_left
                    new_copy_indices = copy_indices.copy()
                else:
                    to_add = out_noffspring - num_added
                    out_offspring_indices.extend(offspring_left[:to_add])
                    new_copy_indices.extend(copy_indices[idx] for idx in offspring_left[:to_add])
                    
            new_offspring = [offspring_solutions[i] for i in  out_offspring_indices]
            variator.evolve(custom_type, parent_solutions.copy(), new_offspring, variable_index, new_copy_indices, **kwargs)
            return set(out_offspring_indices)
        
        # out_nparents < nparents
        if out_noffspring == noffspring:
            parent_map = {}
            new_parents = []
            new_copy_indices = copy_indices.copy()
            added = 0
            for i, parent_index in enumerate(copy_indices):
                if parent_index is None:
                    continue
                if parent_index in parent_map:
                    new_copy_indices[i] = parent_map[parent_index]
                    continue
                
                parent_map[parent_index] = added
                new_copy_indices[i] = added
                new_parents.append(parent_solutions[parent_index])
                added += 1
                
                if added == out_nparents:
                    if i < noffspring - 1: # check remaining for unused copy_indices
                        for j in range(i + 1, noffspring):
                            prev_parent_idx = new_copy_indices[j]
                            if new_copy_indices is not None and prev_parent_idx in parent_map:
                                new_copy_indices[j] = parent_map[prev_parent_idx]
                            else:
                                new_copy_indices[j] = None
                    break
            
            if not added: # all unlabeled
                variator.evolve(custom_type, parent_solutions[:out_nparents].copy(), offspring_solutions.copy(), variable_index, new_copy_indices, **kwargs)
                return set(range(noffspring))
            if added < out_nparents:
                uncopied= np.setdiff1d(np.arange(nparents), list(parent_map.keys()), assume_unique=True)
                new_parents.extend(parent_solutions[i] for i in uncopied[:out_nparents-added])
                assert len(new_parents == out_nparents)

            variator.evolve(custom_type, new_parents, offspring_solutions.copy(), variable_index, new_copy_indices, **kwargs)
            return set(range(noffspring))
        
        # out_noffspring < noffspring and out_nparents < nparents
        return tools._first_variator_subset_case(
            variator, custom_type,
            parent_solutions, offspring_solutions,
            copy_indices, variable_index,
            out_nparents, out_noffspring,
            nparents, noffspring, **kwargs
        )
    
    def _first_mutation(
        self, 
        variator: LocalMutator, custom_type, 
        parents, offspring, variable_index, copy_indices: list, 
        noffspring, strategy, **kwargs):
        """ Assumes variator is a mutator and do_mutation is True. 
        Assumes no alteration of offspring has occured'""" 
        altered_offspring = None
        if strategy == 'all':  
            variator.evolve(custom_type, parents, offspring, variable_index, copy_indices, **kwargs)
            altered_offspring = set(range(noffspring))
        elif strategy == 'rand':
            rand_offspring_idx = np.random.randint(noffspring)
            variator.mutate(custom_type, offspring[rand_offspring_idx], variable_index, **kwargs)
            altered_offspring = {rand_offspring_idx} 
        else: # alter offspring from unique parents
            altered_offspring = set()
            max_noffspring = max(1, noffspring - self.contains_swap)
            seen_parents = set()
            added = 0
            for i in range(noffspring):
                copied_from = copy_indices[i]
                if copy_indices[i] is None:
                    altered_offspring.add(i)
                elif not copied_from in seen_parents:
                    altered_offspring.add(i)
                    seen_parents.add(copied_from)
                else:
                    continue
                added += 1
                if added == max_noffspring:
                    break
            variator.evolve(custom_type, parents, [offspring[i] for i in altered_offspring], variable_index, copy_indices, **kwargs)
        return altered_offspring
    
    def _mutation_only(
        self, 
        custom_type, parent_solutions, offspring_solutions, variable_index, copy_indices, 
        nvariators: int, noffspring: int, start_idx, **kwargs):
        """Used when there are no crossover variators, or when custom_type.do_evolution is False.
        No requirement updating parents"""
        
        multi = self.mutation_selection is None
        curr_idx = start_idx
        if not multi and self.mutation_selection != 'rand':
            while True:
                variator: LocalVariator = self.variators[curr_idx]
                if isinstance(variator, LocalMutator):
                    variator.evolve(custom_type, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs)
                curr_idx = (curr_idx + 1) % nvariators
                if curr_idx == start_idx:
                    break
            return
            
        previously_altered = set()
        while True:
            variator: LocalVariator = self.variators[curr_idx]
            if isinstance(variator, LocalMutator):
                strategy = 'rand' if not multi else self.offspring_selection[curr_idx]
                if strategy == 'rand':
                    rand_offspring_idx = np.random.randint(noffspring)
                    variator.mutate(custom_type, offspring_solutions[rand_offspring_idx], variable_index, **kwargs)
                    if multi:
                        previously_altered = {rand_offspring_idx}
                elif strategy == 'all' or not previously_altered or len(previously_altered) == noffspring:
                    variator.evolve(custom_type, parent_solutions, offspring_solutions, variable_index, copy_indices, **kwargs)
                    if multi:
                        previously_altered = set(range(noffspring))
                else:
                    variator.evolve(custom_type, parent_solutions, [offspring_solutions[i] for i in previously_altered], variable_index, copy_indices, **kwargs)

            curr_idx = (curr_idx + 1) % nvariators
            if curr_idx == start_idx:
                break




