# GOL with gaussian "growth function" - tuned for GOL but can be arbitrary
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

from scipy.signal import convolve2d
import numpy as np


def gaussian_growth_function(x, mean=1, sd=1):
    grid = (1 / np.sqrt(2 * np.pi * sd)) * np.exp((-(x - mean)**2) / (2 * sd**2))
    return grid


# Declare world variables
MAP_SIZE = 201
M = 0.1
N = 1
GAUSS_MEAN = ((5/2) * N) + M
GROWTH_THRESHOLD = (1 / np.sqrt(2 * np.pi)) * np.exp(-(N**2)/8)
kernel = np.array([[N, N, N],[N, M, N],[N, N, N]])
np.random.seed(825)
grid = np.random.randint(0,2,(MAP_SIZE, MAP_SIZE))

print(GAUSS_MEAN)
print(GROWTH_THRESHOLD)

# Setup window to show images
app = pg.mkQApp()
win = QtWidgets.QMainWindow()
win.setWindowTitle('GOL with Scipy')
win.resize(800,800)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()

def iterate(start_matrix, edges="fill"):
    """Convolve over grid then test for "alive" values"""
    counts = convolve2d(start_matrix, kernel, boundary=edges, mode="same")
    ## Basic thresholding (yay, my maths works!!!)
    # new_grid = ((counts >= 2.1) & (counts <= 3.1)).astype(np.uint8)
    ## Gaussian thresholding
    growth_value = gaussian_growth_function(counts, mean = GAUSS_MEAN)
    new_grid = (growth_value >= GROWTH_THRESHOLD).astype(np.uint8)
    return new_grid

# Run Indefinitely
def update():
    global grid
    grid = iterate(grid)
    imv.setImage(grid.T)
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)
app.exec_()