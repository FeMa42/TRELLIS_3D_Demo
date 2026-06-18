import numpy as np
from trellis.utils.printability_utils import fill_occupancy_holes

def test_fills_interior_cavity_3d():
    occ = np.ones((9, 9, 9), dtype=bool)
    occ[4, 4, 4] = False                      # one interior void
    out = fill_occupancy_holes(occ)
    assert out.dtype == bool
    assert out[4, 4, 4] == True               # void filled
    assert out.sum() == 9 * 9 * 9

def test_leaves_open_surface_untouched():
    occ = np.zeros((9, 9, 9), dtype=bool)
    occ[1:8, 1:8, 0:4] = True                 # solid block, open to a face
    out = fill_occupancy_holes(occ)
    assert out.sum() == occ.sum()             # nothing enclosed -> no change

def test_batched_volume():
    occ = np.ones((2, 1, 9, 9, 9), dtype=bool)
    occ[0, 0, 4, 4, 4] = False
    out = fill_occupancy_holes(occ)
    assert out.shape == occ.shape
    assert out[0, 0, 4, 4, 4] == True
