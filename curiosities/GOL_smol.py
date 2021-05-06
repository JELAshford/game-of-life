from itertools import product
import numpy as np

MAP_SIZE = 20
MAP = np.zeros((MAP_SIZE+2, MAP_SIZE+2))
MAP[1:MAP_SIZE+1, 1:MAP_SIZE+1] = np.random.randint(0, 2, (MAP_SIZE, MAP_SIZE))

while True:
    print(MAP)
    NEW_MAP = np.zeros((MAP_SIZE+2, MAP_SIZE+2))
    for (i, j) in product(range(1, MAP_SIZE+1), repeat=2):
        n = np.sum(MAP[i-1:i+2, j-1:j+2])-MAP[i][j]
        if (n == 3) or (MAP[i][j] and n == 2): NEW_MAP[i][j] = 1
    MAP = NEW_MAP
