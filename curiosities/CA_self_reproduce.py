# Visualising a Self Replicating CA
from scipy.signal import convolve2d
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np

# Setup start configuration
MAP_SIZE = 101
mainMat = np.zeros(shape=(MAP_SIZE, MAP_SIZE))
mainMat[49:52, 49:52] = [
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

def iterate(start_matrix):
    return convolve2d(start_matrix, [[1, 1, 1], [1, 0, 1], [1, 1, 1]], boundary="fill", mode="same") % 2

# # Run Indefinitely
# def update():
#     global mainMat
#     mainMat = iterate(mainMat)
#     imv.setImage(mainMat)
# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(50)

# Run for x frames and then view
frames = []
for _ in range(100):
    frames.append(mainMat.T.copy())
    mainMat = iterate(mainMat)
data = np.array(frames)
imv.setImage(data, xvals=np.arange(data.shape[0]+1))

if __name__ == '__main__':
    app.exec_()