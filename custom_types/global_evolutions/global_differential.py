import numpy as np
from ..core import CustomType, GlobalEvolution
from ._generic_type_tools import _get_2D_numpy_integer_array
from ..integer_methods.integer_methods import multi_int_crossover
from ..real_methods.numba_differential import differential_evolve
from platypus import Solution, Integer, Real

class GlobalDifferential(GlobalEvolution):
    """
    Perform differntial evolution on all variable types in a Problem. 
        
    This class also supports evolving generic Platypus Real and Integer types.
        Integer types will be evolved with a crossover of bits
    """
    def __init__(
        self, 
        global_crossover_probability: float | None = None,
        ignore_generics = False,
        generic_step_size: float = 0.25,
        generic_crossover_rate: float = 0.25):
        """ 
        Args:
            global_crossover_probability (float | None, optional): A global "dampening" of crossover. Must be None or > 0. Defaults to None.
                - If None (or >= 1), then CustomType variable's will always have their 'evolve' method called (those methods usually have their own probability of evolution)
                -  To simulate global, non-uniform variation, this value could be decreased over generations with an Algorithm's *nfe* attribute
                    - This behavior would have to be implemented outside of this class 
            ignore_generics (bool, optional): Whether to ignore generic Platypus Real and Integer types during crossover. Defaults to False.
                - If False, this class will evolve those generic types as well (other Platypus types are not supported at this time)
                - If True, only CustomType variables will be evolved.
            generic_step_size: Controls the distribution of evolutions for Platypus Reals. Defaults to 0.25.
            generic_crossover_rate: The crossover rate for Platypus Reals and Integers types. Defaults to 0.25
        """        
        
        self.generic_crossover_rate = 0
        if not ignore_generics:
            self.generic_crossover_rate = generic_crossover_rate
            ignore_generics = self.generic_crossover_rate <= 0
        
        super(GlobalDifferential, self).__init__(4, 1, ignore_generics)
        self.global_crossover_probability = 1 if global_crossover_probability is None or global_crossover_probability >= 1 else global_crossover_probability 
        assert self.global_crossover_probability > 0, f"The global crossover probabilty must be greater than 0 (or no crossover will occur). Got {self.global_crossover_probability}"
        self.generic_step_size = np.float32(generic_step_size) if not ignore_generics else None
        self.generic_crossover_rate = generic_crossover_rate if not ignore_generics else None

    def evolve(self, parents: list[Solution]):
        copy_indices = [0]
        offspring = self.get_parent_deepcopies(parents, 0)
        bit_matrix = None
        jrand = np.random.randint(parents[0].problem.nvars)
        for var_type, variable_index in self.generate_types_to_evolve(offspring[0].problem):
            crossover = variable_index == jrand
            if self.global_crossover_probability < 1 and not (crossover or np.random.uniform() < self.global_crossover_probability):
                continue
            if isinstance(var_type, CustomType):
                var_type.execute_variator(parents, offspring, variable_index, copy_indices, crossover = crossover)
            elif isinstance(var_type, Integer) and (crossover or np.random.uniform() < self.generic_crossover_rate):
                if bit_matrix is None:
                    bit_matrix = _get_2D_numpy_integer_array(parents[1:], parents[0].problem, variable_index)
                new_bits = multi_int_crossover(bit_matrix, 1)[0]
                offspring[0].variables[variable_index] = new_bits.tolist()
                offspring[0].evaluated = False
            elif isinstance(var_type, Real) and (crossover or np.random.uniform() < self.generic_crossover_rate):
                new_real = float(differential_evolve(
                    var_type.min_value, 
                    var_type.max_value,
                    parents[1].variables[variable_index],
                    parents[2].variables[variable_index],
                    parents[3].variables[variable_index],
                    self.generic_step_size,
                    normalize_initial=False))
                offspring[0].variables[variable_index] = new_real
                offspring[0].evaluated = False
                
        return offspring
        