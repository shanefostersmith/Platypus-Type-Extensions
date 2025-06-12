
# import numpy as np
# from ..core import CustomType, GlobalMutation, Placeholder
# from ..integer_methods import int_mutation
# from ..real_methods.numba_differential import _real_mutation
# from platypus import Problem, Solution, Integer, Real

# class MixedMutation(GlobalMutation):
#     """
#     Perform mutation on all variable types in a Problem. 
           
#     This class also supports evolving generic Platypus Real and Integer types.
    
#         - Integer types will be mutated with a bit flip mutation
        
#         - Real types will be mutated with a polynomial mutation
    
#     (This class assumes CustomTypes define one mutation, or indicate which type of mutation to perform internally)
#     """
#     def __init__(
#         self, 
#         problem: Problem, 
#         global_mutation_probability: float | None = None,
#         generic_mutation_probability: float = 0.5,
#         ignore_generics = False,
#         distribution_index: None | float = None,
#         generic_only_distribution_index: bool = False,
#         ):
#         """ `
#         Args:
#             problem (Problem): The Problem class for the optimization.
            
#                 - The Problem class is not stored as an attribute, but used to check CustomTypes and generic types
                
#                 - **The Problem class should not add variables or change variable types after initializing `MixedMutation`**
#                     (i.e. initialize MixedMutation after completing initialization of Problem class)
            
#             global_mutation_probability (_type_, optional): A global "dampening" of mutation. Must be None or > 0. Defaults to None.

#                 - If None (or >= 1), then CustomType variable's will always have their 'mutate' method called (those methods usually have their own probability of evolution)
                
#                 -  To simulate global, non-uniform mutation, this value could be decreased over generations with an Algorithm's *nfe* attribute
#                     - This behavior would have to be implemented outside of this class 
            
#             generic_mutation_probability: The probability that a Playtpus Real or Integer type will be mutated. Not relevant if *ignore_generics* is True or no generic types exist in the Problem
                
#             ignore_generics (bool, optional): Whether to ignore generic Platypus Real and Integer types during mutation. Defaults to False.
#                 If False, this class will evolve those generic types as well (other Platypus types are not supported at this time)
                
#                 If True, only CustomType variables will be evolved.
            
#             distribution_index: Controls the output distribution of float objects during mutation.  Defaults to None.
#                 Larger values produce values closer to parent values
            
#             generic_only_distribution_index: If True, the *distribution_index* parameter is used only for Platypus Real types. If False, the *distribution_index* is also passed to CustomTypes. Defaults to False
#                 Note that, CustomTypes and the Real method also have default values (usually 20). 
                  
            
#         """      
#         self.generic_mutation_probability = 0
#         if not ignore_generics:
#             self.generic_mutation_probability = generic_mutation_probability
#             ignore_generics = self.generic_mutation_probability <= 0
        
#         super(MixedMutation, self).__init__(problem, ignore_generics)
        
#         self.global_mutation_probability = 1 if global_mutation_probability is None or global_mutation_probability >= 1 else global_mutation_probability
#         assert(self.global_mutation_probability > 0), "global_mutation_probability must be greater than 0, or no mutation will occur"
        
#         self.generic_only_distribution_index = generic_only_distribution_index
#         self.distribution_index = np.float64(distribution_index or 20) if generic_only_distribution_index else generic_only_distribution_index
#         if self.distribution_index and not generic_only_distribution_index:
#             self.distribution_index = np.float64(self.distribution_index)
    
#     def mutate(self, parent):
#         offspring = self.get_parent_copy_with_avoid(parent)
#         params = {}
#         if not self.generic_only_distribution_index and self.distribution_index:
#             params = {"distribution_index": self.distribution_index}
        
#         for offspring_variable, var_type, var_idx in self.generate_variables_to_mutate(offspring):
#             if not (self.global_mutation_probability == 1 or np.random.uniform() < self.global_mutation_probability):
#                 continue
            
#             mutation_occured = False
#             if isinstance(var_type, CustomType):
#                 mutated_var, mutation_occured = var_type.mutate(offspring_variable, **params)
#             elif isinstance(var_type, Integer) and (self.generic_mutation_probability == 1 or np.random.uniform() < self.generic_mutation_probability):
#                 bit_array = np.array(offspring_variable, np.bool_)
#                 mutated_var, mutation_occured = int_mutation(bit_array, True).to_list()
#             elif isinstance(var_type, Real) and (self.generic_mutation_probability == 1 or np.random.uniform() < self.generic_mutation_probability):
#                 x = np.float64(offspring_variable)
#                 lb = np.float64(var_type.min_value)
#                 ub = np.float64(var_type.max_value)
#                 distrib_idx = np.float64(self.distribution_index or 20)
#                 mutated_var  = float(_real_mutation(x, lb, ub, distrib_idx))
#                 mutation_occured = mutated_var != offspring_variable
            
#             if mutation_occured:
#                 offspring.variables[var_idx] = mutated_var  
#                 offspring.evaluated = False
        
#         return offspring