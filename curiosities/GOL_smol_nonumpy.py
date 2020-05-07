from itertools import product

MAP_SIZE = 20
MAP = [[0]*(MAP_SIZE+2) for _ in range(MAP_SIZE+2)]
MAP[10][5:15] = [1] * 10

while True:
    NEW_MAP = [[0]*(MAP_SIZE+2) for _ in range(MAP_SIZE+2)]
    for (i, j) in product(range(1, MAP_SIZE+1), repeat=2):
        n = sum([MAP[i + x][j + y] for (x,y) in product(range(-1, 2), repeat=2)])-MAP[i][j]
        if (n == 3) or (MAP[i][j] and n == 2): NEW_MAP[i][j] = 1
    MAP = NEW_MAP
