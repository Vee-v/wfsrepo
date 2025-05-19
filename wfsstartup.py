import numpy as np
import os
import matplotlib.pyplot as plt
from wfstools import *
from astropy.visualization import hist
from scipy.stats import norm
from pathlib import Path

exp_time = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subap_positions = calculate_subaperture_positions(grid_size=11)


def startup(cam):
    """
    Initializes the camera, sets parameters, starts a CameraThread, and computes reference positions.
    Returns reference_positions and the CameraThread instance.
    """

    # 1 Set parameters for the WFS
    set_camera_parameters(cam, t_exp=exp_time)
    # 2 Start camera thread
    camera_thread = CameraThread(cam)
    camera_thread.start()
    # 3 Correct for rotational misalignment with the frames from the async thread
    frames = grab_frames_to_array(cam, 10000, camera_thread=camera_thread)
    thetas = np.array([calculate_rotational_misalignment(frame, cam) for frame in frames])
    theta = np.nanmean(thetas)
    theta_err = np.nanstd(thetas)
    print(f"Rotational misalignment = {np.rad2deg(theta)} +- {np.rad2deg(theta_err)}")
    plt.figure()
    hist(np.rad2deg(thetas), bins="blocks", histtype='stepfilled', alpha=0.2, density=True, label="Binned data")
    mu, std = norm.fit(np.rad2deg(thetas))
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    if not os.path.exists("calib"):
        os.makedirs("calib")
    plt.plot(x, p, 'k', linewidth=2, label="Fit results: mu = %.5f°,  std = %.5f°" % (mu, std))
    plt.legend()
    plt.xlabel("Angle (°)")
    plt.ylabel("Number of measurements")
    plt.savefig(Path("calib") / Path("rotational_misalignment.png"))
    plt.show()
    reference_positions = calculate_reference(subap_positions, theta)
    # 3.5 correct for tiptilt misalignment
    deltas = calculate_mla_tt_misalignment(torch.from_numpy(frames).to(device, dtype=torch.float32), reference_positions) # microns
    reference_positions = calculate_reference(subap_positions, theta, deltas)
    print(f"Final tiptilt misalignment = X {torch.rad2deg(torch.arctan(deltas[0] / 13800))}, Y {torch.rad2deg(torch.arctan(deltas[1] / 13800))} degrees")
    print(f"Final tiptilt misalignment = X {deltas[0]}, Y {deltas[1]} microns")
    # 4 Subtract the intrinsic aberrations to the reference positions
    reference_positions -= torch.from_numpy(mla_intr_shift).to(device, dtype=torch.float32)
    # 5 Get valid subaperture mask
    img = torch.from_numpy(frames.mean(axis=0)).to(device, dtype=torch.float32).squeeze()
    subaps = split_wfs_image(img)
    # Estimate noise baseline (e.g., from a dark frame or the lowest 5% of all pixels)
    noise_baseline = torch.quantile(subaps, 0.05)
    print(f"Noise baseline: {noise_baseline}")
    valid_subaps_mask = get_valid_subaps_mask(subaps, noise_baseline)


    return reference_positions, camera_thread, valid_subaps_mask


