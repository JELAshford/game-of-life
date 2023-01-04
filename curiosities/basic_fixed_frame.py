# GOL with gaussian "growth function" - tuned for GOL but can be arbitrary
from scipy.signal import convolve2d
import pyqtgraph as pg
import numpy as np

# Declare world variables
MAP_SIZE = 200
FRAMES = 1000
kernel = np.array([[1, 1, 1],[1, 7, 1],[1, 1, 1]])
grid = np.random.randint(0,2,(MAP_SIZE, MAP_SIZE))

# Generate and Visualise
FRAME_STORE = [grid]
for _ in range(FRAMES):
    counts = convolve2d(grid, kernel, boundary="fill", mode="same")
    grid = np.isin(counts, [9, 10, 3]).astype(np.uint8)
    FRAME_STORE.append(grid)
FRAME_STORE = np.array(FRAME_STORE)
pg.image(FRAME_STORE)
pg.exec()
