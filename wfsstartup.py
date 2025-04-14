from wfstools import *
from astropy.visualization import hist
from scipy.stats import norm
from tqdm import tqdm


exp_time = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subap_positions = calculate_subaperture_positions(grid_size=11)


def startup():
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            # 1 Set parameters for the WFS
            set_camera_parameters(cam, t_exp=exp_time)
            # 2 Grab frames to correct for rotation mla
            frames = grab_frames_to_array(cam, 5000)
            thetas = np.array([calculate_rotational_misalignment(frame, cam) for frame in frames])
            thetas = thetas[~np.isnan(thetas)]
            theta = np.nanmean(thetas)
            theta_err = np.nanstd(thetas)
            plt.figure()
            hist(np.rad2deg(thetas), bins="blocks", histtype='stepfilled', alpha=0.2, density=True, label="Binned data")
            mu, std = norm.fit(np.rad2deg(thetas))
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label="Fit results: mu = %.5f°,  std = %.5f°" % (mu, std))
            plt.legend()
            plt.xlabel("Angle (°)")
            plt.ylabel("Number of measurements")
            plt.show()
            print(f"Final rotational misalignment = {np.rad2deg(theta)} +- {np.rad2deg(theta_err)}")
            reference_positions = calculate_reference(subap_positions, theta)
    return reference_positions


