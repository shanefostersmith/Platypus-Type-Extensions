from numpy import random
from ..core import GlobalEvolution, LocalVariator
from platypus import Solution, Problem
from typing import Literal

class BasicGlobalEvolution(GlobalEvolution):
    """A basic GlobalEvolution
    
    This class only evolves CustomType objects, and does not specify a standard crossover or deepcopying method"""
    
    def __init__(
        self, 
        nparents: int,
        noffspring: int,
        copy_method: int | Literal['sample'] | Literal['rand'] = 'rand',
        global_dampener: float | None = None,
        
    ):
        """
        Args:
            nparents (int): nparents (int): Number of parent solutions to crossover. Must be >= 1
                - Should be compatible with the LocalVariators of the CustomTypes (warnings will show otherwise)
            noffspring (int): Number of offspring solutions to produce per evolution. Must be > 0
                - Should be compatible with the LocalVariators of the CustomTypes
            copy_method (int | Literal[&#39;sample&#39;] | Literal[&#39;rand&#39;]: Specify a method of choosing parent solutions to deepcopy to create offspring
                - If an `int` is provided, then all deepcopies come from that parent index (must be < nparents)
                - If 'sample' and `nparents >= noffspring`, then parent solutions are chosen at random without replacement
                - If 'rand', then parent solutions are chosen at random with replacement
            global_dampener (float | None, optional): A global "dampening" of crossover and mutation - ie. a probability that any variable's LocalVariator will called. Must be None or > 0. Defaults to None.
                - If None or >= 1, then no dampening will occur.
                - Simple non-uniform crossover/mutation can be achieved by lowering this value over generation. This functionality would have to implemented outside this class.
            
        """ 
        
        self.global_dampener = 1 if global_dampener is None or global_dampener >= 1 else global_dampener
        if self.global_dampener <= 0:
            raise ValueError("The global dampener value cannot be <= 0")
        self.copy_method = copy_method
        if isinstance(copy_method, int):
            if copy_method >= nparents:
                raise ValueError("The 'copy_method' integer is >= nparents")
        elif not isinstance(copy_method, str) or not copy_method in ('rand', 'sample'):
            raise ValueError(f"Invalid 'copy_method' {copy_method} of type {type(copy_method)}")
        super().__init__(nparents, noffspring)

    def evolve(self, parents: list[Solution]):
        # print(f'evolve: arity {self.arity}, {len(parents)}')
        assert isinstance(parents, list)
        copy_indices = None
        if isinstance(self.copy_method, int):
            copy_indices = [self.copy_method for _ in range(self.noffspring)]
        elif self.copy_method == 'sample' and self.arity >= self.noffspring:
            copy_indices = list(range(self.arity)) if self.arity == self.noffspring else random.choice(self.arity, size = self.noffspring, replace = False).tolist()
        else:
            copy_indices = random.choice(self.arity, size = self.noffspring).tolist()
            
        offspring = self.get_parent_deepcopies(parents, copy_indices)
        for custom_type, variable_index in self.generate_types_to_evolve(offspring[0].problem):
            if self.global_dampener == 1 or random.uniform() < self.global_dampener:
                custom_type.execute_variator(parents, offspring, variable_index, copy_indices)
                # print(f"offspring flag: {offspring[0].evaluated}, id {id(offspring) % 100}")
        return offspring
        