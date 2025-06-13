import numpy as np
from ..core import CustomType, GlobalEvolution
from ..global_evolutions._generic_type_tools import real_pcx_evolve, integer_parent_crossover
from platypus import Problem, Solution, Integer, Real

class GlobalPCX(GlobalEvolution):
    """
    Perform parent-centric evolution on all variable types in a Problem. 
        
    For CustomTypes that have the "pcx" option (most do in this library), the real values will be evolved using that method.
        CustomTypes without "pcx", or CustomTypes that do not evolve real values, their default method will be used instead.
        
    This class also supports evolving generic Platypus Real and Integer types.
        Integer types will be evolved using a parent-centric crossover of bits
    """
    def __init__(
        self, 
        problem: Problem, 
        nparents: int,
        noffspring: int,
        global_crossover_probability: float | None = None,
        ignore_generics =False,
        eta: None | float = None,
        zeta: None | float = None,
        generic_only_eta_zeta: bool = False):
        """ `
        Args:
            problem (Problem): The Problem class for the optimization.
            
                - The Problem class is not stored as an attribute, but used to check CustomTypes and generic types
                
                - **The Problem class should not add variables or change variable types after initializing GlobalPCX**
                    (i.e. initialize `GlobalPCX` after completing initialization of Problem class)
                
            nparents (int): Number of parent solutions to crossover. Must be > 1
            
            noffspring (int): Number of offspring solutions to produce per evolution. Must be > 0
            
            global_crossover_probability (_type_, optional): A global "dampening" of crossover. Must be None or > 0. Defaults to None.

                - If None (or >= 1), then CustomType variable's will always have their 'evolve' method called (those methods usually have their own probability of evolution)
                
                -  To simulate global, non-uniform variation, this value could be decreased over generations with an Algorithms "nfe" attribute
                    - This behavior would have to be implemented outside of this class 
                
            ignore_generics (bool, optional): Whether to ignore generic Platypus Real and Integer types during crossover. Defaults to False.
                If False, this class will evolve those generic types as well (other Platypus types are not supported at this time)
                
                If True, only CustomType variables will be evolved.
            
            eta (None | float, optional): Parameter for PCX (see below for more details).  Defaults to None.
                
            zeta (None | float, optional): Parameter for PCX (see below for more details). Defaults to None.
            
            generic_only_eta_zeta: If True, the `eta` and `zeta` parameters are used only for Platypus Real types. If False, these parameters are also passed to CustomTypes. Defaults to False
                Note that, the Real method also has default values.
        
        Note on `eta` and `zeta`:
            An offspring is created by deepcopying a parent;
                
                - Generally, the `eta` parameter is a bias for those deepcopied values and the 'zeta' parameter is a bias for all other parent values. 
                    (in reality, the bias is for the "direction" of those values)
            
                - Regardless of bias, larger `eta` and `zeta` values produce more varied offspring
            
            - If an eta and/or zeta parameter is set here, that value will override CustomTypes' set "eta" and "zeta" attributes.
                (assuming *generic_only_eta_zeta* is False and *ignore_generics* is False)
                
            - If set to None (or *generic_only_eta_zeta* is True), CustomTypes with "eta" and "zeta" parameters will use those their own attributes or a default value (the default is usually 0.1 for both values)
            
            - It generally recommended that `eta` and `zeta` are in the exclusive range (0,1)
        """        
        
        assert nparents > 1, f"There must be at least 2 parents, got {nparents}"
        assert noffspring > 0, f"There must be at least 1 offspring, got {noffspring}"
        super(GlobalPCX, self).__init__(problem, nparents, noffspring, ignore_generics)
        self.global_crossover_probability = 1 if global_crossover_probability is None or global_crossover_probability >= 1 else global_crossover_probability 
        assert self.global_crossover_probability > 0, f"The global crossover probabilty must be greater than 0 (or no crossover will occur)"
        
        self.noffspring = noffspring
        self.eta = eta
        self.zeta = zeta
        self.generic_only_eta_zeta = generic_only_eta_zeta
    
    def evolve(self, parents: list[Solution]):
        offspring = []
        original_indices = np.empty(self.noffspring)
        params = {"real_method": "pcx"} 
        if not self.generic_only_eta_zeta and (self.eta or self.zeta):
            params.update({"eta": self.eta, "zeta": self.zeta})
            
        for i in range(self.noffspring): # Get random deepcopies of parents
            rand_parent_idx = np.random.randint(self.arity)
            original_indices[i] = rand_parent_idx
            offspring.append(self.get_parent_copies(parents[rand_parent_idx]))
        
        real_indices = None
        int_indices = None
        for offspring_vars, var_type, var_idx in self.generate_variables_to_evolve(offspring):
            if self.global_crossover_probability < 1 and not (np.random.uniform() < self.global_crossover_probability):
                continue
            if isinstance(var_type, CustomType):
                evolved_vars = var_type.evolve(parents, var_idx, offspring_vars, **params)
                for i, (new_var, evolved) in enumerate(evolved_vars):
                    curr_offspring: Solution = offspring[i]
                    if evolved:
                        curr_offspring.variables[i] = new_var
                        curr_offspring.evaluated = False
            elif isinstance(var_type, Integer):
                if real_indices is None:
                    int_indices = [var_idx]
                else:
                    int_indices.append(var_idx)
            elif isinstance(var_type, Real):
                if real_indices is None:
                    real_indices = [var_idx]
                else:
                    real_indices.append(var_idx)

        if real_indices:
            real_pcx_evolve(
                parents, offspring, 
                self.eta, self.zeta,
                real_indices=real_indices, 
                original_parent_indices=original_indices)
        if int_indices:
            integer_parent_crossover(
                parents, offspring, 
                int_indices = int_indices
            )
        
        if self._avoid_copy_indices: # BETA: Replace Placeholders
            for i in range(self.noffspring):
                self.replace_all_placeholder_variables(parents[original_indices[i]], offspring[i])
                
        return offspring

