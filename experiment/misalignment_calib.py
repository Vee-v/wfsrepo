import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.animation import FuncAnimation
from functools import partial
from vmbpy import VmbSystem
from vmbpy.camera import Camera
from vmbpy import PixelFormat
from torchvision.transforms import v2
from scipy.optimize import curve_fit
from scipy.stats import norm
from astropy.visualization import hist
from tqdm import tqdm



def set_camera_parameters(camera: Camera, t_exp=None):
    with camera:
        ''' 
        According to the manual, the order of parameter setting needs to be the following (higher parameters affect lower parameters):
            1. Pixel format (MONO8)
            2. Sensor bit depth (MONO8)
            3. Black level (0)
            4. Reverse X/Y (not reverse)
            5. Binning (2x2)
            6. ROI settings (not sure yet)
            7. Exposure time and framerate (619fps should be fastest?)
        '''
        # Set Pixel format and bit depth
        camera.set_pixel_format(PixelFormat.Mono8)

        # Set Black level
        camera.BlackLevel.set(0.)
        
        # Set binning
        camera.BinningHorizontal.set(2)  # Set horizontal binning to 2
        camera.BinningVertical.set(2)    # Set vertical binning to 2

        # Set ROI (Region of Interest)
        camera.Width.set(312)
        camera.Height.set(312)
        camera.OffsetX.set(64)  # 816/2=408; 408-304=104; 104/2=52
        camera.OffsetY.set(0)   # 624/2=312; 312-304=8; 8/2=4

        # Set exposure time
        if t_exp is None:
            camera.ExposureTime.set(500)  # in microseconds
        else:
            camera.ExposureTime.set(t_exp)

        # Set gamma
        camera.Gamma.set(1.)

        # Set gain
        camera.Gain.set(1)
    
def grab_frame(cam):
    frame = cam.get_frame().as_numpy_ndarray()
    return frame

def center_of_gravity_np(img):
 
    y_coords, x_coords = np.repeat(np.arange(28), 28).reshape(28, 28), np.transpose(np.repeat(np.arange(28), 28).reshape(28, 28))
    total_weight = np.sum(img)
    # aux= x_coords * img
    # print(aux.shape, np.sum(aux) / total_weight)
    # plt.imshow(aux.squeeze())
    # plt.show()
    x_centroid = np.sum((img * x_coords)) / total_weight
    y_centroid = np.sum((img * y_coords)) / total_weight

    return [x_centroid, y_centroid]

def center_of_gravity(img):
    """
    Centroids of an image, or tensor of images.
    Centroids over the last 2 dimensions.

    Parameters:
        img (Tensor): (B, W, H)

    Returns:
        Tensor: Tensor of centroid values (B, 2)

    """
    # Ensure image is a 3D tensor
    assert len(img.shape) == 3, "Image should be 2D (B, W, H)"

    x_coords, y_coords = torch.arange(28).view(1, 1, 28).to(img.device), torch.arange(28, dtype=torch.float32).view(1, 28, 1).to(img.device)

    total_weight = img.sum(dim=[1, 2], keepdim=True)
    total_weight = torch.where(total_weight == 0, torch.tensor(float('nan')).to(img.device), total_weight)

    x_centroid = (img * x_coords).sum(dim=[1, 2], keepdim=True) / total_weight
    y_centroid = (img * y_coords).sum(dim=[1, 2], keepdim=True) / total_weight

    return torch.cat([x_centroid, y_centroid], dim=-1).squeeze()  # Shape: [B, 2]

def calculate_rotational_misalignment(img, cam):
    assert len(img.shape) == 2, "Image should be 2D (W, H)"

    cog_y = np.array([center_of_gravity_np(img[2+28*i:28*(i+1)+2, 28*5+2:28*6+2]) for i in range(11)])
    cog_x = np.array([center_of_gravity_np(img[28*5+2:28*6+2, 2+28*i:28*(i+1)+2]) for i in range(11)])
    pitchs = np.array([28*i for i in [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]])

    cog_y[:, :] -= cog_y[5, :]
    cog_y[:, 1] += pitchs[:]

    cog_x[:, :] -= cog_x[5, :] 
    cog_x[:, 0] += pitchs[:]

    y = np.mean(np.vstack((cog_y[:, 0]*-1, cog_x[:, 1])), axis=0)
    yerr = np.std(np.vstack((cog_y[:, 0]*-1, cog_x[:, 1])), axis=0)
    yerr[5]=1e-9
    x = np.mean(np.vstack((cog_y[:, 1], cog_x[:, 0])), axis=0)

    def line(x, a):
        return a * x

    popt, pcov = curve_fit(line, x, y, sigma=yerr, p0=np.deg2rad(0.02))
    if pcov.squeeze() == np.inf:
        return calculate_rotational_misalignment(grab_frame(cam).squeeze(), cam)
    # plt.figure()
    # plt.scatter(cog_x[:, 0], cog_x[:, 1], label="horizontal")
    # plt.scatter(cog_y[:, 1]*-1, cog_y[:, 0], label="vertical")
    # plt.errorbar(x=x, y=y, yerr=yerr, fmt="o")
    # dom = np.linspace(x.min(), x.max(), 50)
    # plt.plot(dom, line(dom, *popt))
    # plt.legend()
    # plt.show()
    # print(f"Rotational misalignment = {np.rad2deg(np.arctan(popt[0]))} degrees")
    return np.arctan(popt[0])

def calculate_mla_tt_misalignment(imgs, reference_positions):
    centroids = torch.zeros(len(imgs), 121, 2, device=device)
    for i, img in enumerate(imgs):
        subaps = split_wfs_image(img)
        centroids[i, :, :] = center_of_gravity(subaps)
    centroids = torch.mean(centroids, axis=0)
    slopes = centroids_to_slopes(centroids, reference_positions)
    # print(slopes.shape)
    return torch.sin(torch.mean(slopes, axis=0))*13800 # in microns  

def split_wfs_image(img):
    """
    Splits an image into its subapertures.
    Creates an view of a the input tensor. (121, 28, 28)

    Returns:
        Tensor: images of each subaperture.
    
    """

    img = v2.CenterCrop(308)(img)
    subaps = img.unfold(0, 28, 28).unfold(1, 28, 28)
    return subaps.contiguous().view(-1, 28, 28)

def centroids_to_slopes(centroids, reference):
    slopes = (centroids - reference) * 18 / 13800  # radian
    return torch.arcsin(slopes)

def calculate_subaperture_positions(grid_size=11, subap_size=28):
    """
    Calculates the x, y pixel coordinates of all subapertures in the Shack-Hartmann grid.

    Args:
        grid_size (int): Number of subapertures along one dimension (e.g., 11x11 grid).
        subap_size (int): Size of each subaperture in pixels.

    Returns:
        Tensor: (grid_size^2, 2) array of x, y positions for each subaperture.
    """
    subap_centers = torch.arange(0, grid_size * subap_size, subap_size, device=device) + (subap_size / 2)-.5
    x, y = torch.meshgrid(subap_centers, subap_centers, indexing='xy')
    return torch.stack((x.flatten(), y.flatten()), dim=1)  # Shape: (grid_size^2, 2)

def calculate_distances_from_center(subap_positions):
    """
    Calculates the distances of each subaperture from the center.

    Args:
        subap_positions (Tensor): (N^2, 2) array of x, y positions of subapertures.

    Returns:
        Tensor: Distances of each subaperture from the center.
    """
    center = subap_positions.mean(dim=0)  # Find the center of the grid
    distances = torch.sqrt(((subap_positions - center) ** 2).sum(dim=1))
    return distances

def calculate_reference(subap_positions, theta, deltas=torch.zeros(1, dtype=torch.float32)):
    """
    Calculates slopes by comparing centroids to rotated expected positions.

    Args:
        centroids (Tensor): Measured centroid positions (N^2, 2).
        subap_positions (Tensor): Subaperture positions in the Shack-Hartmann grid (N^2, 2).
        theta (float): Rotation angle of the lenslet array in radians.

    Returns:
        Tensor: Slopes in radians.
    """
    # Rotate subaperture positions
    cos_theta, sin_theta = np.cos(-theta), np.sin(-theta)
    rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], device=device, dtype=torch.float32)
    rotated_positions = (subap_positions @ rotation_matrix.T)
    # Calculate the expected centroids after accounting for rotation
    reference_centroids = rotated_positions - subap_positions + deltas.to(device)/18. + 13.5

    return reference_centroids

def hudgin_slopes(slopes):  # optimizar
    edge_slopes = list(range(10, 121, 11))
    slopes_x = slopes[:, 0]
    slopes_y = slopes[:-11, 1]
    mask = torch.ones(slopes.size(0), dtype=torch.bool)
    mask[edge_slopes] = False
    slopes_x = slopes_x[mask]
    return slopes_x, slopes_y

def generate_B_matrix(N, piston=False):
    """
    Generates the B matrix for Hudgin geometry for a square grid of N x N phase points.
    
    Args:
        N (int): Number of phase points along one dimension (N x N grid)
        
    Returns:
        torch.Tensor: The B matrix with shape (2 * N * (N - 1), N * N)
    """

    # Total number of phase points
    num_phase_points = N * N
    
    # Total number of slopes in x and y directions
    num_slopes = 2 * N * (N - 1)
    
    # Initialize B matrix (2 * N * (N - 1) rows, N * N columns)
    B = torch.zeros((num_slopes, num_phase_points), device=device)
    
    row = 0
    
    # Loop through each phase point for x-slopes
    for i in range(N):
        for j in range(N-1):  # One fewer slope than phase points in each row
            idx_1 = i * N + j        # Index of phi(i, j)
            idx_2 = i * N + (j + 1)  # Index of phi(i, j+1)
            B[row, idx_1] = 1        # Coefficient for phi(i, j)
            B[row, idx_2] = -1       # Coefficient for phi(i, j+1)
            row += 1
    
    # Loop through each phase point for y-slopes
    for i in range(N-1):  # One fewer slope than phase points in each column
        for j in range(N):
            idx_1 = i * N + j        # Index of phi(i, j)
            idx_2 = (i + 1) * N + j  # Index of phi(i+1, j)
            B[row, idx_1] = 1        # Coefficient for phi(i, j)
            B[row, idx_2] = -1       # Coefficient for phi(i+1, j)
            row += 1
    if not piston:
        piston_row = torch.ones((1, num_phase_points), device=device)
        B = torch.cat((B, piston_row), dim=0)
    return B

def update(i, image, r_phase, slopes_fig, tiptilt, cam, cbar1, cbar2, B, reference_positions):
    frame = grab_frame(cam)
    # image.set_data(np.log(frame+1))
    # image.set_clim(vmin=np.log(frame+1).min(), vmax=np.log(frame+1).max())
    image.set_data(frame)
    image.set_clim(vmin=frame.min(), vmax=frame.max())
    cbar1.update_normal(image)
    img = torch.from_numpy(frame).to(device).squeeze()
    subaps = split_wfs_image(img)
    centroids = center_of_gravity(subaps)
    slopes = centroids_to_slopes(centroids, reference_positions)
    slopes_x, slopes_y = hudgin_slopes(slopes)
    phase = (torch.linalg.lstsq(B, torch.cat([slopes_x, slopes_y, torch.zeros(1, device=device)])).solution / (2*np.pi)) * 633e-3   # radians to micrometers
    r_phase.set_data(phase.reshape(11, 11).to('cpu'))
    r_phase.set_clim(vmin=phase.min().item(), vmax=phase.max().item())
    cbar2.update_normal(r_phase)
    # print(phase.max().item() - phase.min().item())
    slopes_plot, slopes_ax = slopes_fig
    slopes_data = torch.hstack([torch.sin(slopes_x)*13800, torch.sin(slopes_y)*13800]).to("cpu")
    slopes_plot.set_data(np.arange(slopes_x.size(0)*2), slopes_data)
    slopes_ax.set_ylim((slopes_data.min(), slopes_data.max()))
    tiptilt_plot, central_tt, tiptilt_ax = tiptilt
    tt = torch.sin(torch.mean(slopes, axis=0).to("cpu")) * 13800 # microns
    tiptilt_plot[0].set_height(tt[0])
    tiptilt_plot[1].set_height(tt[1])
    central_tt[0].set_height((centroids[60, 0].to("cpu")-13.5)*18)
    central_tt[1].set_height((centroids[60, 1].to("cpu")-13.5)*18)
    aux = torch.abs(tt).max()
    tiptilt_ax.set_ylim((-aux*1.1, aux*1.1))

def plot_wavefront(cam, frame, phase, slopes, slopes_x, slopes_y, centroids, reference_positions):
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2, width_ratios=[1,1])
        ax1 = plt.subplot((gs[0]))
        image = ax1.imshow(frame)
        cbar1 = fig.colorbar(image, ax=ax1, label='ADUs', fraction=0.046)
        # Plot the grid
        ax1.axvline(155, color='red')
        ax1.axvline(155 + 1, color='red')
        ax1.axhline(155, color='red')
        ax1.axhline(155 + 1, color='red') 
        ax2 = plt.subplot(gs[1])
        r_phase = ax2.imshow(phase.reshape(11, 11).to('cpu'), cmap='plasma')
        cbar2 = fig.colorbar(r_phase, ax=ax2, label='microns', fraction=0.046)
        ax3 = plt.subplot(gs[2])
        slopes_plot, = ax3.plot(torch.cat([torch.sin(slopes_x)*13800, torch.sin(slopes_y)*13800]).to('cpu'))
        ax3.grid()
        ax3.set_xlabel("X slope and Y slope")
        ax3.set_ylabel("P-V of local wavefront (microns)")
        ax4 = plt.subplot(gs[3])
        tiptilt_plot = ax4.bar(["X", "Y"], torch.mean(slopes, axis=0).to("cpu")*torch.tensor(13800), label="tip-tilt")
        central_tt = ax4.bar(["X", "Y"], (centroids[60, :].to("cpu")-13.5) * torch.tensor(18), width=0.5, color="red", label="central subap tip-tilt")
        ax4.set_ylabel("Microns")
        ax4.legend()
        ax4.grid()
        ani = FuncAnimation(fig, partial(update, image=image, r_phase=r_phase, slopes_fig=(slopes_plot, ax3), tiptilt=(tiptilt_plot, central_tt, ax4), cam=cam,
                                            cbar1=cbar1, cbar2=cbar2, B=B, reference_positions=reference_positions), interval=0, cache_frame_data=False)
        plt.show()

def review_alignment(theta=None, theta_err=None, t_exp=500):
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            # set parameters for wavefront sensor cmos
            set_camera_parameters(cam, t_exp=t_exp)
            if theta is None or theta_err is None:
                # grab a frame to correct for rotation mla
                thetas = np.array([calculate_rotational_misalignment(grab_frame(cam).squeeze(), cam) for i in tqdm(range(5000))])
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
                return theta, theta_err
            if not theta is None:
                reference_positions = calculate_reference(subap_positions, theta)
                print("\nPlease center the middle subaperture (red bars --> 0).\n")
                frame = grab_frame(cam).squeeze()   
                img = torch.from_numpy(frame).to(device, dtype=torch.float32).squeeze()
                subaps = split_wfs_image(img)
                centroids = center_of_gravity(subaps)
                slopes = centroids_to_slopes(centroids, reference_positions)
                slopes_x, slopes_y = hudgin_slopes(slopes)
                phase = (torch.linalg.lstsq(B, torch.cat([slopes_x, slopes_y, torch.zeros(1, device=device)])).solution / (2*np.pi)) * 633e-3   # radians to micrometers
                plot_wavefront(cam, frame, phase, slopes, slopes_x, slopes_y, centroids, reference_positions)
            # grab frames to correct for tiptilt mla
            imgs = [torch.from_numpy(grab_frame(cam).squeeze()) for i in tqdm(range(1000))]
            reference_positions = calculate_reference(subap_positions, theta)
            deltas = calculate_mla_tt_misalignment(imgs, reference_positions) # microns
            reference_positions = calculate_reference(subap_positions, theta, deltas)
            print(f"Final tiptilt misalignment = X {torch.rad2deg(torch.arctan(deltas[0] / 13800))}, Y {torch.rad2deg(torch.arctan(deltas[1] / 13800))} degrees")
            print(f"Final tiptilt misalignment = X {deltas[0]}, Y {deltas[1]} microns")
            # measure
            frame = grab_frame(cam).squeeze()   
            img = torch.from_numpy(frame).to(device, dtype=torch.float32).squeeze()
            subaps = split_wfs_image(img)
            centroids = center_of_gravity(subaps)
            slopes = centroids_to_slopes(centroids, reference_positions)
            slopes_x, slopes_y = hudgin_slopes(slopes)
            phase = (torch.linalg.lstsq(B, torch.cat([slopes_x, slopes_y, torch.zeros(1, device=device)])).solution / (2*np.pi)) * 633e-3   # radians to micrometers
            plot_wavefront(cam, frame, phase, slopes, slopes_x, slopes_y, centroids, reference_positions)
            return deltas
        

# def main_saved_image():
#     frame = np.load("wfs-frame.npy")
#     fig = plt.figure()
#     gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
#     ax1 = plt.subplot((gs[0]))
#     image = ax1.imshow(frame)
#     cbar1 = fig.colorbar(image, ax=ax1, label='ADUs', fraction=0.046)
#     img = torch.from_numpy(frame).to(device, dtype=torch.float32).squeeze()
#     subaps = split_wfs_image(img)
#     centroids = center_of_gravity(subaps)
#     theta = calculate_misalignment(frame)
#     reference_positions = calculate_reference(subap_positions, theta)
#     # print(reference_positions)
#     slopes = centroids_to_slopes(centroids, reference_positions)
#     slopes_x, slopes_y = hudgin_slopes(slopes)
#     phase = torch.linalg.lstsq(B, torch.cat([slopes_x, slopes_y, torch.zeros(1, device=device)])).solution * 500
#     ax2 = plt.subplot(gs[1])
#     r_phase = ax2.imshow(phase.reshape(11, 11).to('cpu'), cmap='plasma')
#     cbar2 = fig.colorbar(r_phase, ax=ax2, label='microns', fraction=0.046)
#     ax3 = plt.subplot(gs[2])
#     slopes_plot, = ax3.plot(torch.cat([torch.sin(slopes_x)*500/18, torch.sin(slopes_y)*500/18]).to('cpu'))
#     ax3.set_xlabel("X slope and Y slope")
#     ax3.set_ylabel("P-V of local wavefront (microns)")
#     plt.show()


### 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subap_positions = calculate_subaperture_positions(grid_size=11)
B = generate_B_matrix(11)

###

def main():
    rot = input("Do you need to calibrate rotations? y/n\n")
    if rot != "n":
        t_exp = float(input("Exposure time in microseconds?\n"))
        theta, theta_err = review_alignment(t_exp=t_exp)
        np.save("theta.npy", theta)
        np.save("theta_err.npy", theta_err)
    else:
        theta = np.load("theta.npy")
        theta_err = np.load("theta_err.npy")
    d = float(input("Distance in mm?"))
    deltas = review_alignment(theta, theta_err)
    np.save(f"deltas_{d:.3f}.npy", deltas.to('cpu').numpy())


if __name__ == "__main__":
    main()