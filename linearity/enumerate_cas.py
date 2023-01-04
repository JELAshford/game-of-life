## Let's have a think about how many possible CA's we've got outselves in for here
from itertools import combinations
from rule_conversion import *

from tqdm import tqdm
import numpy as np 

# ## Make my own
# test_disc = DiscreteRuleset(
#     kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
#     AA = np.array([2, 3]),
#     DD = np.array([1, 3, 4, 6, 7, 8]),
#     AD = np.array([1, 4, 7, 8, 9]),
#     DA = np.array([2, 5]),
# )
# test_kernel = discrete_to_kernelcounts(test_disc)
# print(test_kernel)

# ## Try running it?
# run_with_rules(test_kernel)

# AA and AD must contain all numbers up to sum of kernel, and be mutually exclusive (disjoint)
# DA and DA must also obey this criteria


base_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
all_vals = set(np.arange(np.sum(base_kernel)+1))

num_pair_sets = 2**len(all_vals) - 2 # inspired by (((2**len(all_vals))-1)/2)+1

# Let's do this manually; create all combinations of disjoint pairs
all_combos = []
for set_len in np.arange(1, len(all_vals)):
    for set1 in combinations(all_vals, set_len):
        set2_base = all_vals.difference(set1)
        for set2 in combinations(set2_base, len(set2_base)):
            all_combos.append([np.array(set1), np.array(set2)])
assert len(all_combos) == 2**len(all_vals) - 2

# Okay, so if that's the number of options for AA and AD, the number
# of total CAs must be: len(all_combos) ** 2. Oh dear, that's quite
# a few!!

# Perhaps we can filter these a bit with my linear theory! Or at 
# least let's see if we can extract 10 non-linear and 10 linear
# rule sets

# # We can iterate over the combos, and extract the linear sep check
# for combo1 in tqdm(all_combos):
#     for combo2 in tqdm(all_combos):
#         # Create ruleset
#         disc_rule = DiscreteRuleset(
#             kernel = base_kernel,
#             AA = combo1[0],
#             DD = combo2[0],
#             AD = combo1[1],
#             DA = combo2[1]
#         )
        
#         # # Convert to kernel
#         # kern_rule = discrete_to_kernelcounts(disc_rule)
#         # if kern_rule.linear_sep:
#         #     print("Found!")
        
#         # Test lineraity
#         linear, (edge, mid, alive) = test_linearity(disc_rule)
#         if linear:
#             with open("linear_models.txt", "a") as logfile:
#                 logfile.write(f"E:{edge}, M:{mid}, c1:{combo1}, c2:{combo2}, alive: {set(alive)}\n")

# Extract one from the list
lin_kernel = KernelCountRuleset(
    kernel = np.array([[1, 1, 1], [1, -1, 1], [1, 1, 1]]),
    A = np.array([8, 7, 6])
)
run_with_rules(lin_kernel)