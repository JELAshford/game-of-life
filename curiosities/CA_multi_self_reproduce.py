# Visualising a Self Replicating CA
from scipy.signal import convolve2d
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np

# Setup start configuration
MAP_SIZE = 201
grid = np.zeros(shape=(MAP_SIZE, MAP_SIZE, 3))
mid = int(MAP_SIZE/2)
l, h = mid-1, mid+2
grid[l:h, l:h, 1] = [
    [1, 0, 1],
    [0, 0, 0],
    [1, 0, 1]
]
grid[l:h, l:h, 2] = [
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
]
grid[l:h, l:h, 0] = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]

# Setup window to show images
app = pg.mkQApp()
win = QtGui.QMainWindow()
win.setWindowTitle('Self-Reproducing CA')
win.resize(800,800)
imv = pg.ImageView()
imv.ui.roiBtn.hide()
imv.ui.menuBtn.hide()
win.setCentralWidget(imv)
win.show()

def iterate(start_grid):
    for dim in range(3):
        start_grid[:, :, dim] = convolve2d(
            start_grid[:, :, dim], 
            [
                [1, 1, 1], 
                [1, 0, 1], 
                [1, 1, 1]
            ], 
            boundary="fill", 
            mode="same"
        ) % 200
    return start_grid

# Run Indefinitely
# def update():
#     global grid
#     grid = iterate(grid)
#     imv.setImage(grid)
# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(100)

# Run for x frames and then view
frames = []
for _ in range(300):
    frames.append(grid.copy())
    grid = iterate(grid)
data = np.array(frames)
imv.setImage(data, xvals=np.arange(data.shape[0]+1))

if __name__ == '__main__':
    app.exec_()