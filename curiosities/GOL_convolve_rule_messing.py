# Conway's Game of Life - lets experiment different rulesets with the conv
from scipy.signal import convolve2d
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np

# My rationale is we can mess around with the conv-matrix and target values
# targets = [9, 10, 3]
# conv_matrix = [
#     [1, 2, 1],
#     [1, 4, 2],
#     [1, 2, 1]
# ] # FROGGER

# targets = [9, 10, 3]
# conv_matrix = [
#     [2, 1, 2],
#     [1, 4, 1],
#     [2, 1, 2]
# ] # INTERSECTION

# targets = [9, 10, 3]
# conv_matrix = [
#     [1, 2, 1],
#     [2, 4, 2],
#     [1, 2, 1]
# ] # INTERSECTION PILEUP

targets = [9, 14, 4]
conv_matrix = [
    [1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 7, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1]
] # PERSISTENT SLIMY

# targets = [9, 14, 4] + list(range(15, 25))
# conv_matrix = [
#     [0, 1, 1, 1, 0],
#     [1, 1, 1, 1, 1],
#     [1, 1, 7, 1, 1],
#     [1, 1, 1, 1, 1],
#     [0, 1, 1, 1, 0]
# ] # SPACE FILLING SLIMY BOIIS

# targets = [9, 14, 4] + list(range(16, 23))
# conv_matrix = [
#     [1, 1, 1, 1, 1],
#     [1, 0, 1, 0, 1],
#     [1, 1, 7, 1, 1],
#     [1, 0, 1, 0, 1],
#     [1, 1, 1, 1, 1]
# ] # SOLID WITH SLIMY OUTLINE

# targets = [9, 14, 4] + list(range(16, 30))
# conv_matrix = [
#     [1, 1, 1, 1, 1],
#     [1, 0, 1, 0, 1],
#     [1, 1, 7, 1, 1],
#     [1, 0, 1, 0, 1],
#     [1, 1, 1, 1, 1]
# ] # THE ORB

# Declare world variables
MAP_SIZE = 400
mid = int(MAP_SIZE/2)
# grid = np.random.randint(0,2,(MAP_SIZE, MAP_SIZE))

inset = 100
grid = np.zeros((MAP_SIZE, MAP_SIZE))
grid[mid-inset:mid+inset, mid-inset:mid+inset] = np.random.randint(0,2,(inset*2, inset*2))

# Setup window to show images
app = pg.mkQApp()
win = QtGui.QMainWindow()
win.setWindowTitle('GOL with Scipy')
win.resize(800,800)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()

def iterate(start_matrix, edges="fill"):
    """Convolve over grid then test for "alive" values"""
    counts = convolve2d(start_matrix, conv_matrix, boundary=edges, mode="same")
    return np.isin(counts, targets)

# Run Indefinitely
def update():
    global grid
    grid = iterate(grid)
    # grid[mid-10:mid+10, mid-10:mid+10] = 1
    imv.setImage(grid.T, autoRange=False)
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)

if __name__ == '__main__':
    app.exec_()
