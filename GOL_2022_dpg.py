from scipy.signal import convolve2d
from einops import repeat
import numpy as np
import array

import dearpygui.dearpygui as dpg
dpg.create_context()


def gol_step(start_matrix, edges="fill"):
    """Convolve over grid then test for "alive" values"""
    counts = convolve2d(start_matrix, [[1, 1, 1], [1, 7, 1], [
                        1, 1, 1]], boundary=edges, mode="same")
    return np.isin(counts, [9, 10, 3]).astype(np.uint8)


def update_raw_texture():
    # Get current texture data, and convert to 2d slice of 3d space
    state_view = np.frombuffer(state, dtype=np.float32).view().reshape(
        GOL_SIZE, GOL_SIZE, 4)[:, :, 0]
    # Run GOL on state
    state_view = gol_step(state_view)
    # Reshape to texture format
    state_view = np.repeat(state_view, 4)
    state[:] = array.array('f', state_view)


def update_raw_texture_nocopy():
    # Get current texture data, and convert to 3d space
    state_view = np.frombuffer(state, dtype=np.float32).view().reshape(
        GOL_SIZE, GOL_SIZE, 4)
    # Run GOL on 2d slice of state (all positions are equal)
    tmp = state_view[:, :, 0]
    tmp = gol_step(tmp)
    state_view[:] = repeat(tmp, 'h w -> h w c', c=4)


GOL_SIZE = 1000
PAD = int(GOL_SIZE*0.1)
INDENT = int(PAD/2)
WINDOW_SIZE = int(GOL_SIZE + PAD)

# Define start texture
state = np.repeat(np.random.randint(0, 2, size=(GOL_SIZE, GOL_SIZE)), 4)
state = array.array('f', state)
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(width=GOL_SIZE, height=GOL_SIZE,
                        format=dpg.mvFormat_Float_rgba, default_value=state, tag="gol_texture")


with dpg.window(tag="MainWindow"):
    dpg.add_image("gol_texture", pos=(INDENT, INDENT))


dpg.create_viewport(title='GOL', width=WINDOW_SIZE, height=WINDOW_SIZE)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("MainWindow", True)
while dpg.is_dearpygui_running():
    update_raw_texture_nocopy()
    dpg.render_dearpygui_frame()
dpg.destroy_context()
