# Platypus-Type-Extensions

##Types:

1. **Categorical List**
    1. Example: ("category_1", "category_2", ..., "category_n")
    2. Returns one of the categories
   
2. **Real List**
    1.  Example: (0.25, 1, 1.25, 1.75, 4, 6, 6.33)
    2.  Returns one of the numbers
       
3. **Real Step**
    1. Example: (0.25, 4), step = 0.25
    2. Returns a number in: 0.25 + int(x) * step (where 0 <= x <= 15)
         
4. **Discrete Growth/Decay Distribution**
    1. Given a strictly increasing or decreasing distribution function(s), returns a sorted set of 'y' values from an evenly spaced, random number of 'x' values.
    2. If more than one distribution function is provided, one of these distributions will be chosen randomly.
    3. Note, a 'distribution function' must include a function 'f(x) -> y', an inverse function 'f^(-1)(y) -> x', a minimum 'x' value, a maximum 'x' value and maximum number of points. 
    
5. **Fixed-Size Weight Bins**
    1. Given a number of 'total features' and a fixed 'subset size', this returns a 'weight_directory' (an array) in the following format:
        1. The features are labeled 0 to 'total_features - 1'.
        2. The weights are labeled 0 to 'subset_size'.
        3. A random number of 'subset_size' features are chosen and weight bins are randomly assigned to each of the 'active' features.
            1. If a feature 'i' is not part of the current subset, then 'weight_directory[i] = -1'.
            2. If a feature 'i' is a part of the current subset, then 'weight_directory[i] = random_weight'.
    2. User may specify a minimum and maximum value for a weight bin (usually in range (0, 1]).
         
6. **Variable-Size Weight Bins**
    1. Given a number of 'total features', a minimum 'subset size' and a maximum 'subset set', this returns a tuple '(random_subset, random_weights)' in the following format:
        1. The features are labeled 0 to 'total_features - 1'.
        2. The weight associated with feature 'random_subset[i]' is 'random_weights[i]'
    2. User may specify a minimum and maximum value for a weight bin, and the minimum and maximum subset size.
