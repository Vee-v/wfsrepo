import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import block_diag
from uncertainties import ufloat




def polynomial_basis(x, y, degree):
    basis = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            basis.append((x**i) * (y**j))
    return np.array(basis)

def polynomial_derivative_x(x, y, degree):
    basis_dx = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            basis_dx.append(i * (x**(i-1)) * (y**j) if i > 0 else 0)
    return np.array(basis_dx)

def polynomial_derivative_y(x, y, degree):
    basis_dy = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            basis_dy.append(j * (x**i) * (y**(j-1)) if j > 0 else 0)
    return np.array(basis_dy)

def second_derivative_x(coeffs, x, y, degree):
    """Calculate second derivative w.r.t. x at a point (x, y)."""
    d2_phi_dx2 = 0
    index = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if i >= 2:  # Second derivative only for i >= 2
                d2_phi_dx2 += coeffs[index] * i * (i - 1) * (x**(i - 2)) * (y**j)
            index += 1
    return d2_phi_dx2

def second_derivative_y(coeffs, x, y, degree):
    """Calculate second derivative w.r.t. y at a point (x, y)."""
    d2_phi_dy2 = 0
    index = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if j >= 2:  # Second derivative only for j >= 2
                d2_phi_dy2 += coeffs[index] * j * (j - 1) * (x**i) * (y**(j - 2))
            index += 1
    return d2_phi_dy2

def lsq_fit_wf(slopes_x, slopes_y):
    num_basis = (D + 1) * (D + 2) // 2  # Number of coefficients
    A = np.zeros((2 * N**2, num_basis))  # Design matrix
    b = np.zeros(2 * N**2)  # Target slopes

    # Populate A and b
    for idx, (x, y, sx, sy) in enumerate(zip(x_coords.ravel(), y_coords.ravel(), slopes_x.ravel(), slopes_y.ravel())):
        basis_dx = polynomial_derivative_x(x, y, D)
        basis_dy = polynomial_derivative_y(x, y, D)
        A[2*idx, :] = basis_dx
        A[2*idx+1, :] = basis_dy
        b[2*idx] = sx
        b[2*idx+1] = sy
    coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return coefficients

def reconstruct_wavefront(x, y, coeffs, degree):
    basis = polynomial_basis(x, y, degree)
    return np.dot(coeffs, basis)

def spherical_wavefront_4th_order(coords, Rx, Ry):
    """
    Fourth-order spherical wavefront model.
    coords: Flattened array of (x, y) coordinates.
    Rx, Ry: Radii of curvature in X and Y directions.
    """
    x, y = coords
    z = (x**2) / (2 * Rx) + (y**2) / (2 * Ry) + ((x**2 + y**2)**2) / (8 * (Rx**3)) + ((x**2 + y**2)**2) / (8 * (Ry**3))
    return z

def estimate_wf_curvature(slopes):
    slopes_x, slopes_y = slopes[:, 0], slopes[:, 1]  # radians
    a = lsq_fit_wf(slopes_x, slopes_y)
    for i in range(n):
        for j in range(n):
            wavefront_r[i, j] = reconstruct_wavefront(rx_coords[i, j], ry_coords[i,j], a, D)  / (2*np.pi) * 633e-3 # radians to micrometers
    # Flatten coordinates and wavefront for curve_fit
    x_flat = rx_coords.ravel()
    y_flat = ry_coords.ravel()
    coords_flat = np.vstack((x_flat, y_flat))  # Combine into a single array
    wavefront_flat = wavefront_r.ravel()

    # Initial guess for Rx and Ry
    initial_guess = [1.0, 1.0]

    # Use curve_fit to fit the 4th-order spherical wavefront model
    popt, pcov = curve_fit(
        spherical_wavefront_4th_order,
        coords_flat,
        wavefront_flat,
        p0=initial_guess
    )
    return popt[0]*1e-6, popt[1]*1e-6, pcov*1e-6  # radius of curvature micrometers to meters

def parabola(x, a, b, c):
    return a * x**2 + b * x + c


D = 8  # Degree of polynomial
N = 11  # Number of subapertures
d = np.array([181.200, 183.200, 185.200, 187.200, 389.700, 391.700, 393.700, 395.700, 397.700, 399.700, 401.700, 403.700, 405.700, 407.700, 573.200, 575.200, 577.200, 579.200, 581.200, 583.200], dtype=np.float32)
# oldv3d = np.array([388.200, 393.200, 398.200, 403.200, 408.200, 490.200, 495.200, 500.200, 505.200, 510.200, 563.200, 568.200, 573.200, 578.200, 583.200], dtype=np.float32)  # radius of curvature measurements (R_ref) in millimeters
# d = np.array([328, 494, 631, 754, 879, 880, 1031, 1080, 1104, 1180, 1201, 1307, 1480], dtype=np.int16)
wfroc = np.zeros([len(d), 2])  # measured wavefront radius of curvature [:, x], [:, y]
wfroc_err = np.zeros([len(d), 2]) # measured wavefront radius of curvature standard error [:, x], [:, y]
n = 101 # number of points to reconstruct
x_coords, y_coords = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))  # coords of lsq fit of wavefront
rx_coords, ry_coords = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))  # coords of wavefront reconstruction
wavefront_r = np.zeros((n, n))  # reconstructed wavefront
covariance_m = np.zeros((len(d), 2, 2))

def main():
    # for i in range(9):
    #     d[i] = int(input(f"Input measurement distance in mm {i+1}/9\n--\n"))
    wf = [np.load(f"measurements/slopes_{i:.3f}.npy") for i in d] # radian 
    for i in range(len(d)):
        wfroc[i, 0], wfroc[i, 1], covariance_m[i, :, :] = estimate_wf_curvature(wf[i])  # meters
        wfroc_err[i, 0], wfroc_err[i, 1] = np.sqrt(covariance_m[i, 0, 0]), np.sqrt(covariance_m[i, 1, 1])
    # print(np.mean(wf, axis=0))
    # print(wfroc)
    # return
    wfrho = 1 / wfroc  # 1/meters
    wfrho_err = wfroc_err / wfroc
    rho_ref = 1e3/d  # 1/m
    rho_ref_err = 0.0005/rho_ref
    delta_rho_x = np.abs(wfrho[:, 0] - rho_ref)  # 1/meters
    # print(wfrho[:, 0])
    # print(rho_ref)
    # return
    delta_rho_x_err = wfrho_err[:, 0] + rho_ref_err
    
    popt, pcov = curve_fit(
        parabola,
        rho_ref,
        delta_rho_x*1e-3,
        sigma=delta_rho_x_err*1e-3,  # revisar como propagar una matriz de covarianza entera
        absolute_sigma=True
    )
    sigma = np.sqrt(np.diagonal(pcov))
    a = ufloat(popt[0], sigma[0])
    b = ufloat(popt[1], sigma[1])
    c = ufloat(popt[2], sigma[2])
    print(popt)
    design_focus = 13.8  # mm
    design_pitch = 0.5  # mm
    real_focus = design_focus / (1 + b/1000)  # to mm
    delta_f = real_focus * b/1000
    delta_R = a / (b/1000) / 1000  # to mm
    real_pitch = design_pitch / (1 + c/(1e6) * real_focus) # to mm
    delta_P = np.abs(real_pitch * c/(1e6) * real_focus)  # to mm

    text_res = f"\nX axis\n$f_0$={real_focus:.2u}mm\n$\delta R$={delta_R:.2u}mm\n$P_0$={real_pitch:.7f}mm\n$\delta f$={delta_f:.1u}mm\n$\delta P$={delta_P:.2u}mm"
    domain = np.linspace(np.min(rho_ref)-0.1, np.max(rho_ref)+0.1, 100)
    plt.text(1.6, 0.67, text_res)
    plt.plot(domain, parabola(domain, *popt), color="b", label="X axis")#, label=f"a={a:.5f}, b={b:.5f}, c={c:.5f}", color="b")
    plt.errorbar(rho_ref, delta_rho_x*1e-3, yerr=delta_rho_x_err*1e-3, xerr=rho_ref_err, fmt="sb", capsize=3, label="X curvature data")
    delta_rho_y = np.abs(wfrho[:, 1] - rho_ref)
    delta_rho_y_err = wfrho_err[:, 1] + rho_ref_err
    popt, pcov = curve_fit(
        parabola,
        rho_ref,
        delta_rho_y*1e-3,
        sigma=delta_rho_y_err*1e-3,
        absolute_sigma=True
    )
    sigma = np.sqrt(np.diagonal(pcov))
    a = ufloat(popt[0], sigma[0])
    b = ufloat(popt[1], sigma[1])
    c = ufloat(popt[2], sigma[2]) 
    print(popt)  
    real_focus = design_focus / (1 + b/1000)  # to mm
    delta_f = real_focus * b/1000
    delta_R = a / (b/1000) / 1000  # to mm
    real_pitch = design_pitch / (1 + c/(1e6) * real_focus) # to mm
    delta_P = np.abs(real_pitch * c/(1e6) * real_focus)  # to mm
    text_res = f"\nY axis\n$f_0$={real_focus:.2u}mm\n$\delta R$={delta_R:.2u}mm\n$P_0$={real_pitch:.7f}mm\n$\delta f$={delta_f:.1u}mm\n$\delta P$={delta_P:.2u}mm"
    plt.text(3.5, 0.28, text_res,)
    plt.plot(domain, parabola(domain, *popt), ls="--", color="red", label="Y axis")#, label=r"$\delta R \cdot \dfrac{\delta f}{f_0}$"+f"={a:.3u}, "+ r"$\dfrac{\delta f}{f_0}$"+f"={b:.3u}, c={c:.5f}")
    plt.errorbar(rho_ref, delta_rho_y*1e-3, yerr=delta_rho_y_err*1e-3, xerr=rho_ref_err, fmt="^k", capsize=3, label="Y curvature data")
    plt.ylabel(r"$\delta\rho$ [$10^{-3}$ m$^{-1}$]")
    plt.xlabel(r"$\rho_{ref}$ [m$^{-1}$]")
    # plt.ylim((0, 50))
    # plt.xlim((0, 2.8))
    plt.legend()
    plt.show()
main()