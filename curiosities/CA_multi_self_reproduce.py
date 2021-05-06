# Visualising a Self Replicating CA
from scipy.signal import convolve2d
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np

# Setup start configuration
MAP_SIZE = 101
grid = np.zeros(shape=(MAP_SIZE, MAP_SIZE, 3))
grid[49:52, 49:52, 0] = [
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
win.setCentralWidget(imv)
win.show()

def iterate(start_grid):
    start_grid[:, :, 0] = convolve2d(start_grid[:, :, 0], [[1, 1, 1], [1, 0, 1], [1, 1, 1]], boundary="fill", mode="same") % 2
    return start_grid

# # Run Indefinitely
# def update():
#     global grid
#     grid = iterate(grid)
#     imv.setImage(grid)
# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(50)

# Run for x frames and then view
frames = []
for _ in range(100):
    frames.append(grid.copy())
    grid = iterate(grid)
data = np.array(frames)
data = data.reshape((101, 101, 3, 100))
imv.setImage(data, xvals=np.arange(data.shape[3]+1))

if __name__ == '__main__':
    app.exec_()