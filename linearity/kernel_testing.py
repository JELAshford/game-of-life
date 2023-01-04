# Trying to find an ideal kernel for the convolution which has at the very least 
# a continuous range of "alive" conditions - ideally all of them should be on 
# one side of a filter (i.e. all > N with no dead conditions above). Seperable, 
# the word
import numpy as np

# All EDGE_VALUEs (neighbours) must be the same, but doesn't have to be the same as 
# the mid value.

# Mid/Neighbour behaviour is:
# 1, (2, 3) = ON                OR  0, (3) = ON
# 1, (1, 4, 5, 6, 7, 8) = OFF   OR  0, (1, 2, 4, 5, 6, 7, 8) = OFF
# So need a situation where MID_VALUE + (2, 3) and 3 are distinct from
# MID_VALUE + EDGE_VALUE + (1, 4, 5, 6, 7, 8) and EDGE_VALUE + (1, 2, 4, 5, 6, 7, 8)

MAX_ABS = 10
STEP = 1

for EDGE_VALUE in np.arange(-MAX_ABS, MAX_ABS, STEP):
    for MID_VALUE in np.arange(-MAX_ABS, MAX_ABS, STEP):
        
        on_values = [MID_VALUE + (EDGE_VALUE * 2), MID_VALUE + (EDGE_VALUE * 3), (EDGE_VALUE * 3)]
        off_values = [MID_VALUE + (EDGE_VALUE * v) for v in (1, 4, 5, 6, 7, 8)] + [(EDGE_VALUE * v) for v in [1, 2, 4, 5, 6, 7, 8]]
        
        # Test if results for on don't include off
        overlap = set(off_values).intersection(set(on_values))
        # Test that outcomes are linearly seperable
        linear_sep = all(np.array(on_values) < min(off_values)) | all(np.array(on_values) > max(off_values))
        
        #combine
        test = (not overlap) & linear_sep
        
        if test: 
            print(f"Success! {MID_VALUE} as value for centre, and {EDGE_VALUE} for edge works")
            print(f"Active Values: {set(on_values)}")
            print(f"Inactive Values: {set(off_values)}")
            print("")