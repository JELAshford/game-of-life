# GOL with gaussian "growth function" - visualised with dpg
from einops import repeat, rearrange

import dearpygui.dearpygui as dpg
import threading

from scipy.signal import convolve2d
import numpy as np


def gaussian_growth_function(x, mean=1, sd=1):
    """Calculate the gaussian resonse from a vector of initial x values"""
    grid = (1 / np.sqrt(2 * np.pi * sd)) * \
        np.exp((-(x - mean)**2) / (2 * sd**2))
    return grid


def update_gaussian_plot():
    """Update the plot data using the stored values for mean"""
    mean_value = dpg.get_value("gauss_mean")
    new_ys = gaussian_growth_function(xs, mean=mean_value)
    dpg.set_value("gaussian_plot", [xs, new_ys])


def update_mean_value(sender, app_data, user_data):
    dpg.set_value("gauss_mean", dpg.get_value(sender))
    update_gaussian_plot()


def update_threshold(sender, app_data, user_data):
    dpg.set_value("growth_thresh", dpg.get_value(sender))
    update_gaussian_plot()


def grid_to_image(grid_array):
    float_grid = grid_array.astype(np.float32)
    rgba_grid = repeat(float_grid, 'h w -> h w c', c=4)
    return rearrange(rgba_grid, 'h w c -> (h w c)')


def count_alive_dead(grid_array):
    alive_count = [np.sum(grid_array == 1) / MAP_SIZE**2]
    dead_count = [np.sum(grid_array == 0) / MAP_SIZE**2]
    return alive_count, dead_count


def iterate(start_matrix, edges="fill"):
    """Convolve over grid then test for "alive" values"""
    counts = convolve2d(start_matrix, kernel, boundary=edges, mode="same")
    growth_value = gaussian_growth_function(
        counts, mean=dpg.get_value("gauss_mean"))
    new_grid = (growth_value >= dpg.get_value(
        "growth_thresh")).astype(np.uint8)
    return new_grid


def update_grid(sender, app_data, user_data):
    """Take the image texture for the grid and run one iteration of it,
    dynamically reassigning the texture afterwards"""
    global grid, alive_counts, dead_counts
    grid = iterate(grid)
    # Store values (and calculations)
    alive_counts.append(np.sum(grid == 1) / MAP_SIZE**2)
    dead_counts.append(np.sum(grid == 0) / MAP_SIZE**2)
    dpg.set_value("iteration_count", dpg.get_value("iteration_count")+1)
    # Update plots
    iterations = np.arange(0, dpg.get_value("iteration_count"))
    dpg.set_value("ca_grid", grid_to_image(grid))
    dpg.set_value("alive_plot", [iterations, np.expand_dims(
        np.array(alive_counts), axis=1)])
    dpg.set_value("dead_plot", [iterations, np.expand_dims(
        np.array(dead_counts), axis=1)])
    dpg.fit_axis_data(xax)


def run_grid(id, should_go):
    """Set the grid running at a fixed speed, if on a separate thread should updat independantly"""
    global grid
    while True:
        update_grid(None, None, None)
        if not should_go():
            break
    return None


def begin_grid_run(sender, app_data, user_data):
    # Get the thread running
    dpg.set_value("should_be_running", True)
    grid_run_thread = threading.Thread(name="run_grid", target=run_grid, args=(
        1, lambda: dpg.get_value("should_be_running")), daemon=True)
    grid_run_thread.start()
    # Disable unusable buttons
    dpg.disable_item("step_button")
    dpg.disable_item("reset_button")
    dpg.disable_item("run_button")
    dpg.hide_item("step_button")
    dpg.hide_item("reset_button")
    dpg.hide_item("run_button")


def stop_grid_run(sender, app_data, user_data):
    # Request the thread stops
    dpg.set_value("should_be_running", False)
    # Re-enable usable buttons
    dpg.enable_item("step_button")
    dpg.enable_item("reset_button")
    dpg.enable_item("run_button")
    dpg.show_item("step_button")
    dpg.show_item("reset_button")
    dpg.show_item("run_button")


def reset_grid(sender, app_data, user_data):
    """Reset the global grid to a random state"""
    global grid, alive_counts, dead_counts
    grid = np.random.randint(0, 2, (MAP_SIZE, MAP_SIZE))

    dpg.set_value("ca_grid", grid_to_image(grid))
    dpg.set_value("iteration_count", 1)

    alive_counts, dead_counts = count_alive_dead(grid)
    dpg.set_value("alive_plot", [[1], alive_counts])
    dpg.set_value("dead_plot", [[1], dead_counts])
    dpg.fit_axis_data(xax)


# Map and intiial config for all variables (GOL)
MAP_SIZE = 1001
M = 0.1
N = 1
GAUSS_MEAN = ((5/2) * N) + M
GROWTH_THRESHOLD = (1 / np.sqrt(2 * np.pi)) * np.exp(-(N**2)/8)

# Define the kernel for visualisation and processing
kernel = np.array([[N, N, N], [N, M, N], [N, N, N]])
# kernel = np.array([[N, N, N, N, N],[N, N, M, N, N],[N, N, N, N, N]])
xs = np.linspace(0*N+M, 8*N+M, 100)
ys = gaussian_growth_function(xs, mean=GAUSS_MEAN)

# Create original grid
np.random.seed(825)
grid = np.random.randint(0, 2, (MAP_SIZE, MAP_SIZE))

# Create store for alive/dead counts
alive_counts, dead_counts = count_alive_dead(grid)


# Create the context and begin building DPG Stuff
dpg.create_context()

# Add grid as dynamic texture (allows updating)
with dpg.texture_registry():
    dpg.add_dynamic_texture(
        MAP_SIZE, MAP_SIZE, grid_to_image(grid), tag="ca_grid")

# Add variables to the value registry
with dpg.value_registry():
    dpg.add_bool_value(default_value=False, tag="should_be_running")
    dpg.add_int_value(default_value=1, tag="iteration_count")
    dpg.add_double_value(default_value=GROWTH_THRESHOLD, tag="growth_thresh")
    dpg.add_double_value(default_value=GAUSS_MEAN, tag="gauss_mean")


# Visualise the Gaussian Growth Function
with dpg.window(label="Gaussian Function View", width=400, height=300):
    with dpg.plot(height=-1, width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, label="x")
        dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="y_axis")
        # Add the gaussian line and threshold label
        dpg.add_line_series(x=xs, y=ys, tag="gaussian_plot", parent="y_axis")
        dpg.add_drag_line(label="thresh_hline", vertical=False, color=[
                          255, 0, 0, 255], default_value=dpg.get_value("growth_thresh"), callback=update_threshold)
        dpg.add_drag_line(label="thresh_vline", vertical=True, color=[
                          0, 0, 255, 255], default_value=dpg.get_value("gauss_mean"), callback=update_mean_value)


# Visualise the alive/dead counts over the last N generations
with dpg.window(label="Alive/Dead History", width=400, height=300, pos=[0, 300]):
    with dpg.plot(height=-1, width=-1):
        xax = dpg.add_plot_axis(axis=0, label="Iteration")
        yax = dpg.add_plot_axis(axis=1, label="Count", tag="yaxis_ad")
        dpg.set_axis_limits_auto(xax)
        dpg.set_axis_limits(yax, ymin=0, ymax=1)
        # Draw the lines of alive_counts and dead_counts
        iterations = np.arange(0, dpg.get_value("iteration_count"))
        dpg.add_plot_legend()
        dpg.add_scatter_series(x=iterations, y=alive_counts,
                               parent="yaxis_ad", tag="alive_plot", label="Alive Count")
        dpg.add_scatter_series(x=iterations, y=dead_counts,
                               parent="yaxis_ad", tag="dead_plot", label="Dead Count")


# Visualise the CA State
with dpg.window(label="CA Viewer", width=600, height=600, pos=[400, 0]):
    dpg.add_image("ca_grid", width=500, height=500, pos=[50, 0])
    dpg.add_button(label="Step Sim", width=300,
                   callback=update_grid, tag="step_button", pos=[150, 510])
    dpg.add_button(label="Reset Sim", width=300,
                   callback=reset_grid, tag="reset_button", pos=[150, 540])
    dpg.add_button(label="Run!", width=140, callback=begin_grid_run,
                   tag="run_button", pos=[150, 570])
    dpg.add_button(label="Stop!", width=140, callback=stop_grid_run,
                   tag="stop_button", pos=[310, 570])


# Create window and do all the necessary DPG stuff
dpg.create_viewport(title='Gaussian CA Visualiser', width=1000, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
