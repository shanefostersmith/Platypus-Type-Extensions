from platypus.config import PlatypusConfig
from .core import (
    CustomType, LocalVariator, LocalMutator, 
    LocalCompoundMutator, LocalGAOperator, LocalSequentialOperator,
    GlobalEvolution)
from .global_evolutions.global_differential import GlobalDifferential
from .global_evolutions.general_global_evolution import GeneralGlobalEvolution
from .bins_and_sets.bins_and_sets import (
    SetPartition, WeightedSet,
    WeightedSetMutation, WeightedSetCrossover,
    ActiveBinSwap, FixedSubsetSwap, BinMutation
)
from .lists_and_ranges.lists_ranges import (
    Categories, CategoryCrossover, CategoryMutation,
    RealList, RealListDE, RealListPCX, RealListPM,
    SteppedRange, SteppedRangeCrossover, StepMutation,
    MultiRealRange, MultiIntegerRange, ArrayCrossover,
    MultiDifferentialEvolution, MultiPCX, MultiRealPM, 
    MultiIntegerCrossover, MultiIntegerMutation
)
