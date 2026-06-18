"""Occupancy-domain printability helpers (TRELLIS Stage-1).

Based on the printability-eval investigation: applying scipy.ndimage.binary_fill_holes
to the Stage-1 64^3 occupancy grid (before Stage-2) reduces slicer support volume by
~15% (about -1300 mm^3 median) while preserving fine detail (f1@0.01 ~ 0.80).
"""
import numpy as np
import scipy.ndimage as ndi


def fill_occupancy_holes(occ: np.ndarray) -> np.ndarray:
    """Fill interior voids in a boolean occupancy grid.

    Applies ``scipy.ndimage.binary_fill_holes`` to each trailing (D, H, W) volume.
    Accepts a 3-D array or any leading batch/channel dims, e.g. (N, 1, D, H, W).
    Returns a boolean array of the same shape. Open (surface-connected) cavities
    are left untouched -- only topologically enclosed voids are filled.
    """
    occ = np.asarray(occ, dtype=bool)
    if occ.ndim < 3:
        raise ValueError(f"expected >=3 dims (..., D, H, W), got shape {occ.shape}")
    if occ.ndim == 3:
        return ndi.binary_fill_holes(occ)
    spatial = occ.shape[-3:]
    flat = occ.reshape(-1, *spatial)
    filled = np.empty_like(flat)
    for i in range(flat.shape[0]):
        filled[i] = ndi.binary_fill_holes(flat[i])
    return filled.reshape(occ.shape)
