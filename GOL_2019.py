# Conway's Game of Life - Done better than 19 y/o James
import numpy as np
import pyqtgraph as pg

# Declare world variables
MAP_SIZE = 100
# Populate world randomly with edge buffer of 0
mainMat = np.zeros((MAP_SIZE+2, MAP_SIZE+2))
mainMat[1:MAP_SIZE+1, 1:MAP_SIZE+1] = np.random.randint(0,2,(MAP_SIZE, MAP_SIZE))

# Activate graphics window
window = pg.GraphicsWindow(title='The Game of Life')
gol = window.addPlot(title='')
gol.showGrid(True, True, 1)

def iterate(start_matrix):
    # Create new map of zeros
    end_matrix = np.zeros((MAP_SIZE+2, MAP_SIZE+2))

    # Iterate over all non edge positions
    for i in range(1,MAP_SIZE+1):
        for j in range(1,MAP_SIZE+1):
            # Count active neighbours
            count = np.sum(start_matrix[i-1:i+2, j-1:j+2])-start_matrix[i][j]
            # Perform life logic
            if start_matrix[i][j] == 0:
                if count == 3:
                    end_matrix[i][j] = 1
            elif count == 2 or count == 3:
                end_matrix[i][j] = 1
    return end_matrix

def draw(game_map):
    # Clear map
    gol.clear()
    # Convert Matrix to plot-worthy coordinates
    points = np.argwhere(game_map == 1)
    # Set the data on the plot
    gol.plot(points[:, 0].tolist(), points[:, 1].tolist(), pen=None, s=20., symbolBrush=(255, 255, 0), symbol='o')
    # Process changes
    pg.QtGui.QApplication.processEvents()

while True:
    mainMat = iterate(mainMat)
    draw(mainMat)
