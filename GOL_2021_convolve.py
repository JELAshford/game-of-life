# Conway's Game of Life - Hopefully done better than 21 y/o James
# (100x faster than original grid and set implementation)
from scipy.signal import convolve2d
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np

# Mid/Neighbour behaviour is:
# 1, (2, 3) = ON                OR  0, (3) = ON
# 1, (1, 4, 5, 6, 7, 8) = OFF   OR  0, (1, 2, 4, 5, 6, 7, 8) = OFF
# So need a situation where MID_VALUE + (2, 3) and 3 are distinct from
# MID_VALUE + (1, 4, 5, 6, 7, 8) and (1, 2, 4, 5, 6, 7, 8)
# for MID_VALUE in range(-10, 10):
#     on_values = [MID_VALUE + 2, MID_VALUE + 3, 3]
#     off_values = [MID_VALUE + v for v in (1, 4, 5, 6, 7, 8)] + [1, 2, 4, 5, 6, 7, 8]
#     if not np.any([v in off_values for v in on_values]): 
#         print(f"Succes! {MID_VALUE} as value for centre works")
# Test it out! Let's use 7, I like 9, 10, 3 as the comparison numbers :D

# Declare world variables
MAP_SIZE = 2000
grid = np.random.randint(0,2,(MAP_SIZE, MAP_SIZE))

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
    counts = convolve2d(start_matrix, [[1, 1, 1], [1, 7, 1], [1, 1, 1]], boundary=edges, mode="same")
    return np.isin(counts, [9, 10, 3])

# Run Indefinitely
def update():
    global grid
    grid = iterate(grid)
    imv.setImage(grid)
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)

# Run for x frames and then view
# frames = []; frame_count = 100
# for _ in range(frame_count):
#     frames.append(grid.T.copy())
#     grid = iterate(grid)
# imv.setImage(np.array(frames), xvals=np.arange(frame_count+1))

if __name__ == '__main__':
    app.exec_()
