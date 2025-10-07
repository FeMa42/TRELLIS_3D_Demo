from typing import *
import hashlib
import numpy as np


def get_file_hash(file: str) -> str:
    sha256 = hashlib.sha256()
    # Read the file from the path
    with open(file, "rb") as f:
        # Update the hash with the file content
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

# ===============LOW DISCREPANCY SEQUENCES================

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def generate_views_from_angles(azimuths, elevations, fixed_radius=None):
    """
    Generate camera views from specified azimuth and elevation angles.
    
    Parameters:
    - azimuths: List of azimuth angles in degrees (0-360)
    - elevations: List of elevation angles in degrees (0-180, measured from top)
    - fixed_radius: Optional fixed distance from center for all views. If None, random distances are generated
    
    Returns:
    - List of camera view dictionaries with yaw, pitch, radius, and fov
    """
    if len(azimuths) != len(elevations):
        raise ValueError("Azimuths and elevations must have the same length")
    
    num_views = len(azimuths)
    yaws = []
    pitchs = []
    
    # Convert azimuth and elevation to yaw (phi) and pitch (theta)
    for az, el in zip(azimuths, elevations):
        # Convert azimuth to yaw (phi in radians)
        yaw = az * (np.pi / 180)
        
        # Convert elevation from top to pitch (theta in radians)
        # In the original code, pitch is measured from equator (-π/2 to π/2)
        # But elevation is measured from top (0 to 180)
        pitch = (90 - el) * (np.pi / 180)
        
        yaws.append(yaw)
        pitchs.append(pitch)
    
    if fixed_radius is not None:
        # Use fixed radius for all views
        radius = [fixed_radius] * num_views
        fov = [2 * np.arcsin(np.sqrt(3) / 2 / fixed_radius)] * num_views
    else:
        # Generate random distances as in the original implementation
        fov_min, fov_max = 10, 20
        radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
        radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
        k_min = 1 / radius_max**2
        k_max = 1 / radius_min**2
        ks = np.random.uniform(k_min, k_max, (num_views,))
        radius = [1 / np.sqrt(k) for k in ks]
        fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    
    # Create the views dictionaries
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} 
             for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    return views