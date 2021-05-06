# Conway's Game of Life - Hopefully done better than 21 y/o James
# (10x faster than original grid and set implementation)
from scipy.signal import convolve2d
from itertools import product
import pyqtgraph as pg
import numpy as np

# Declare world variables
MAP_SIZE = 100
mainMat = np.random.randint(0,2,(MAP_SIZE, MAP_SIZE))

# Activate graphics window
window = pg.GraphicsWindow(title='The Game of Life')
gol = window.addPlot(title='')
# gol.showGrid(True, True, 1)

def iterate(start_matrix):
    # Create new map of zeros
    end_matrix = np.zeros((MAP_SIZE, MAP_SIZE))
    # Count neighbours
    counts = convolve2d(start_matrix, [[1, 1, 1], [1, 0, 1], [1, 1, 1]], boundary="fill", mode="same")
    # Apply rules
    for (x, y) in product(range(MAP_SIZE), range(MAP_SIZE)):
        neighbours = counts[y, x]
        alive = start_matrix[y, x]
        if (neighbours == 3) or (alive and neighbours == 2):
            end_matrix[y, x] = 1
    return end_matrix

def draw(game_map):
    # Clear map
    gol.clear()
    # Plot alive points
    points = np.argwhere(game_map == 1)
    gol.plot(points[:, 0].tolist(), points[:, 1].tolist(), pen=None, s=20., symbolBrush=(255, 255, 0), symbol='o')
    # Process changes
    pg.QtGui.QApplication.processEvents()

while True:
    mainMat = iterate(mainMat)
    draw(mainMat)
