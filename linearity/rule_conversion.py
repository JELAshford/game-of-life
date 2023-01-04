"""Convert between rule encodings and run them"""
from scipy.signal import convolve2d

from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

from dataclasses import dataclass
from itertools import product
import numpy as np


# Encoding 1: Discrete
@dataclass
class DiscreteRuleset:
    """Kernel and count maps for Alive (A)/Dead (D) combos"""
    kernel: np.array = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    AA: np.array = np.array([2, 3])
    DD: np.array = np.array([1, 2, 4, 5, 6, 7, 8])
    AD: np.array = np.array([1, 4, 5, 6, 7, 8])
    DA: np.array = np.array([3])


# Encoding 2: Kernel Counts
@dataclass
class KernelCountRuleset:
    kernel: np.array = np.array([[1, 1, 1], [1, 7, 1], [1, 1, 1]])
    A: np.array = np.array([9, 10, 3])
    linear_sep: bool = False



#### CONVERTERS ####

def discrete_to_kernelcounts(discrete : DiscreteRuleset, max_val : int = 10):
    """Convert a discrete ruleset object to a KernelCountRuleset
    object. For now, assumes that the kernel only contains a single weight, 
    but this may be extended in future. 

    Args:
        discrete (DiscreteRuleset): Source rulset
        max_val (int): Maximum value for the kernel search (Default: 10)
    Returns:
        KernelCountRulset: Output ruleset,
    """
    
    # Extract info from kernel (create centre mask and work) out the 
    # multipliers for each positions
    w, h = discrete.kernel.shape
    
    self_kernel = np.zeros(shape=(w, h))
    self_kernel[w//2, h//2] = 1
    
    neighbour_kernel = np.ones(shape=(w, h))
    neighbour_kernel[w//2, h//2] = 0
    
    # Setup ranges of possible EDGE and MID values
    edge_values = np.arange(-max_val, max_val, 1)
    mid_values = np.arange(-max_val, max_val, 1)

    # Iterate combinations, return valid (no overlap between Alive and Dead counts)
    valid_outputs = dict()
    for (edge_val, mid_val) in product(edge_values, mid_values):

        # Construct alive and dead values
        alive = np.hstack(([mid_val + (discrete.AA * edge_val)], [discrete.DA * edge_val]))[0]
        dead = np.hstack(([mid_val + (discrete.AD * edge_val)], [discrete.DD * edge_val]))[0]
        
        # Test overlap between alive/dead counts
        overlap_test = set(alive).intersection(set(dead))
        linear_sep_test = all(np.array(alive) < min(dead)) | all(np.array(alive) > max(dead))
        
        test = (not overlap_test) #& linear_sep_test
        if test: 
            valid_outputs[(edge_val, mid_val)] = [alive, linear_sep_test]
                
    # Choose output with smallest product of edge/mid val
    sorted_out = sorted(valid_outputs, key=lambda x: abs(x[0]) * abs(x[1]))
    best_edge, best_mid = sorted_out[0]
    alive, linear_sep_bool = valid_outputs[(best_edge, best_mid)]

    kernelcounts = KernelCountRuleset(
        kernel = (neighbour_kernel * best_edge) + (self_kernel * best_mid),
        A = alive,
        linear_sep = linear_sep_bool
    )
    return kernelcounts


### TEST LINEARITY ###

def test_linearity(ruleset : DiscreteRuleset, max_val : int = 5):
    
    # Setup ranges of possible EDGE and MID values
    # edge_values = np.arange(-max_val, max_val, 1)
    edge_values = [1]
    mid_values = np.arange(-max_val, max_val, 1)

    # Iterate combinations, return valid (no overlap between Alive and Dead counts)
    for (edge_val, mid_val) in product(edge_values, mid_values):

        # Construct alive and dead values
        alive = np.hstack(([mid_val + (ruleset.AA * edge_val)], [ruleset.DA * edge_val]))[0]
        dead = np.hstack(([mid_val + (ruleset.AD * edge_val)], [ruleset.DD * edge_val]))[0]
        
        # Test overlap between alive/dead counts
        linear_sep_test = all(np.array(alive) < min(dead)) | all(np.array(alive) > max(dead))
        if linear_sep_test:
            return True, (edge_val, mid_val, alive)
    return False, (None, None, None)
        



### RUNNER ####
def run_with_rules(ruleset : KernelCountRuleset):
    global grid
        
    # Declare world variables
    MAP_SIZE = 200
    grid = np.random.randint(0,2,(MAP_SIZE, MAP_SIZE))

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
        counts = convolve2d(start_matrix, ruleset.kernel, boundary=edges, mode="same")
        return np.isin(counts, ruleset.A).astype(np.uint8)

    # Run Indefinitely
    def update():
        global grid
        grid = iterate(grid)
        imv.setImage(grid.T)
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(20)
    app.exec_()




if __name__ == "__main__":
    
    ## Test the default
    dis = DiscreteRuleset()
    print(discrete_to_kernelcounts(dis))
    
    ## Make my own
    test_disc = DiscreteRuleset(
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
        AA = np.array([2, 3]),
        DD = np.array([1, 3, 4, 6, 7, 8]),
        AD = np.array([1, 4, 7, 8, 9]),
        DA = np.array([2, 5]),
    )
    test_kernel = discrete_to_kernelcounts(test_disc)
    print(test_kernel)
    
    ## Try running it?
    run_with_rules(test_kernel)