import numpy as np
from prysm.polynomials.zernike import zernike_nm_der
from numpy.linalg import lstsq

# --- 1. Define subaperture centers (normalized to unit pupil) ---
# Example: 5x5 grid of lenslets
num = 5
x = np.linspace(-1, 1, num)
y = np.linspace(-1, 1, num)
X, Y = np.meshgrid(x, y)
# Only keep points inside unit circle
mask = X**2 + Y**2 <= 1
x_pts = X[mask]
y_pts = Y[mask]
N = len(x_pts)

# --- 2. Choose Zernike modes (Noll indices 1..4 for piston, tip, tilt, defocus) ---
noll_to_nm = {
    1: (0, 0),
    2: (1, -1),
    3: (1, 1),
    4: (2, 0),
}
modes = [1, 2, 3, 4]
M = len(modes)

# Pre-allocate design matrices
A = np.zeros((2 * N, M))

# --- 3. Build design matrix by evaluating Zernike derivatives ---
r = np.hypot(x_pts, y_pts)
theta = np.arctan2(y_pts, x_pts)

for j, jn in enumerate(modes):
    n, m = noll_to_nm[jn]
    # dZ/dr and dZ/dθ from prysm (orthonormalized by default)
    dZ_dr, dZ_dtheta = zernike_nm_der(n, m, r, theta, norm=True)
    # Convert to Cartesian derivatives:
    # dZ/dx = cosθ * dZ/dr - (sinθ / r) * dZ/dθ
    # dZ/dy = sinθ * dZ/dr + (cosθ / r) * dZ/dθ
    dZ_dx = np.cos(theta) * dZ_dr - np.sin(theta) * dZ_dtheta / np.where(r == 0, 1, r)
    dZ_dy = np.sin(theta) * dZ_dr + np.cos(theta) * dZ_dtheta / np.where(r == 0, 1, r)
    # Fill design matrix
    A[:N, j] = dZ_dx
    A[N:, j] = dZ_dy

# --- 4. Simulate measured slopes (for testing) ---
# e.g., true coefficients [0.1, 0.2, -0.1, 0.05]
true_a = np.array([0.1, 0.2, -0.1, 0.05])
slopes_x = A[:N] @ true_a
slopes_y = A[N:] @ true_a
s = np.concatenate([slopes_x, slopes_y])

# --- 5. Solve for Zernike coefficients via least squares ---
estimated_a, residuals, rank, svals = lstsq(A, s, rcond=None)

print("True coeffs:     ", true_a)
print("Estimated coeffs:", estimated_a)