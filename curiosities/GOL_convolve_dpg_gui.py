# Conway's Game of Life - using the speedy convolutional method and DearPyGUI
from scipy.signal import convolve2d
from einops import repeat
import numpy as np

from dearpygui.simple import *
from dearpygui.core import *

def reset_grid(*args):
    grid = get_data("grid_state")
    grid = np.random.randint(0,2,(MAP_SIZE, MAP_SIZE))
    add_data("grid_state", grid)

def update_canvas(*args):
    # global grid
    grid = get_data("grid_state")
    # Update grid with GOL rules
    counts = convolve2d(grid, [[1, 1, 1], [1, 7, 1], [1, 1, 1]], boundary="fill", mode="same")
    grid =  np.isin(counts, [9, 10, 3])
    add_data("grid_state", grid)
    # Convert to texture and draw
    w, h = grid.shape
    img = repeat(grid, 'h w -> (h w c)', c=4)*255
    add_texture("from_array", img, w, h)
    clear_drawing("GOL") # images can't be updated with modify_draw_command()
    draw_image("GOL_Image", "from_array", [EDGE,EDGE], [DRAW_SIZE, DRAW_SIZE])

# Declare world variables
MAP_SIZE = 300
EDGE = 10
DRAW_SIZE = 800
grid = np.random.randint(0,2,(MAP_SIZE, MAP_SIZE))

# Setup DPG window to show GOL drawing at desired size
with window(name="Game of Life"):
    add_drawing("GOL_Image", width=DRAW_SIZE, height=DRAW_SIZE)
    add_data("grid_state", grid)
    add_button(name="Reset", callback=reset_grid, tip="Reset the grid with random start")

# Setup DPG rendering and start
set_render_callback(update_canvas)
set_main_window_size(DRAW_SIZE+(EDGE*2), DRAW_SIZE+(EDGE*2)+100)
set_main_window_title("Game of Life")
start_dearpygui(primary_window="Game of Life")
