# Conway's Game of Life - Hopefully done better than 20 y/o James
# Use an encoding method that keeps a set of active cells in tuple format
import matplotlib.pylab as plt
import numpy as np

def neighbours(pos):
    # Return list of all the neighbour positions
    return(set((pos[0]+dx, pos[1]+dy) for (dx, dy) in ADJ))

def update(active):
    # Get cell tuples that surround active cells
    scan_cells = active.union(*[neighbours(c) for c in active])
    # For each cell of interest, perform the rules and create new cell data
    new_active = set()
    for c in scan_cells:
        # Get number of active adjacent cells
        n = len(neighbours(c).intersection(active))
        # Add to new_active if active by condensed Conway GOL rules
        if (n == 3) or (c in active and n == 2): new_active.add(c)
    return(new_active, scan_cells)

# Visualise grid state
def draw(active, scanned, vis_size=50):
    # Create 2D array to be visualised
    window = [[0]*vis_size for _ in range(vis_size)]
    # Add in all scanned cells in scanned
    for s in scanned:
        if (max(s) < vis_size and min(s) >= 0): window[s[0]][s[1]] = 0.5
    # Add in all active cells in active
    for c in active: 
        if (max(c) < vis_size and min(c) >= 0): window[c[0]][c[1]] = 1
    # Draw to window
    plt.imshow(window, cmap='gray')
    plt.show()

# Define screen size
S = 200; H = int(S/2)
# List of adjacent coordinates
ADJ = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# active_cells = {(H, H) ,(H+2, H-1), (H+2,H), (H+1, H+2), (H+2, H+3), (H+2, H+4), (H+2, H+5)} # Acorn
active_cells = {(x, y) for x in range(1, S-1) for y in range(1, S-1) if np.random.random() > 0.5} #Random Start

for _ in range(1000):
    # Update cells with Conway GOL Rules
    next_active_cells, scanned_cells = update(active_cells)
    # Draw current arrangement and scanned cells
    draw(active_cells, scanned_cells, vis_size=S)
    # Update active cells
    active_cells = next_active_cells
