# %% [markdown]
# # Wavefront sensor algorithm development

# %% [markdown]
# This ipynb is an organized notebook for the development of the wavefront sensing algorithm for the TARdYS front-end AO system.

# %% [markdown]
# ### Context

# %% [markdown]
# There is an idea for using cacao software for the AO control, however that is going to be tackled after solving this through python.

# %% [markdown]
# ### Preamble

# %%
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math
from matplotlib.animation import FuncAnimation
from functools import partial
from vmbpy import VmbSystem
from vmbpy.camera import Camera
from vmbpy import PixelFormat
from torchvision.transforms import v2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
# ### Camera reading

# %%
def set_camera_parameters(camera: Camera):
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
        
        camera.ExposureTime.set(5000)  # in microseconds
        # camera.ExposureTime.set(17.596)

        # Set gamma
        camera.Gamma.set(1.)

        # Set gain
        camera.Gain.set(1)

        
def grab_frame(cam):
    frame = cam.get_frame().as_numpy_ndarray()
    return frame


def update(i, image, r_phase, slopes_plot, cam, cbar1, cbar2, B):
    frame = grab_frame(cam)
    image.set_data(frame)
    image.set_clim(vmin=frame.min(), vmax=frame.max())
    cbar1.update_normal(image)
    img = torch.from_numpy(frame).to(device).squeeze()
    subaps = split_wfs_image(img)
    centroids = center_of_gravity(subaps)
    slopes = centroids_to_slopes(centroids)
    slopes_x, slopes_y = hudgin_slopes(slopes)
    phase = torch.linalg.lstsq(B, torch.cat([slopes_x, slopes_y, torch.zeros(1, device=device)])).solution * 500.
    r_phase.set_data(phase.reshape(11, 11).to('cpu'))
    r_phase.set_clim(vmin=phase.min().item(), vmax=phase.max().item())
    cbar2.update_normal(r_phase)
    print(phase.max().item() - phase.min().item())
    slopes_plot.set_data(np.arange(slopes_x.size(0)*2), torch.cat([slopes_x, slopes_y]).to('cpu'))


# %% [markdown]
# ### Utils

# %%
def center_of_gravity(img):
    """
    Centroids an image, or tensor of images.
    Centroids over the last 2 dimensions.

    Parameters:
        img (Tensor): (B, W, H)

    Returns:
        Tensor: Tensor of centroid values (B, 2)

    """
    # Ensure image is a 3D tensor
    assert len(img.shape) == 3, "Image should be 2D (B, W, H)"

    x_coords, y_coords = torch.arange(28).float().view(1, 1, 28).to(img.device), torch.arange(28).float().view(1, 28, 1).to(img.device)

    total_weight = img.sum(dim=[1, 2], keepdim=True)
    total_weight = torch.where(total_weight == 0, torch.tensor(float('nan')).to(img.device), total_weight)

    x_centroid = (img * x_coords).sum(dim=[1, 2], keepdim=True) / total_weight
    y_centroid = (img * y_coords).sum(dim=[1, 2], keepdim=True) / total_weight

    return torch.cat([x_centroid, y_centroid], dim=-1).squeeze()  # Shape: [B, 2]

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

def centroids_to_slopes(centroids):
    slopes = (centroids - 13.5) / 12.803  # radian
    return slopes

def hudgin_slopes(slopes):  # optimizar
    edge_slopes = list(range(10, 121, 11))
    slopes_x, slopes_y = slopes[:, 0], slopes[:, 1]
    mask = torch.ones(slopes.size(0), dtype=torch.bool)
    mask[edge_slopes] = False
    slopes_x, slopes_y = slopes_x[mask], slopes_y[mask]
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
# %% [markdown]
# ### Main

# %%

def main_video_stream():
    print('Building Geometry Matrix\n')
    B = generate_B_matrix(11)
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            # set parameters for wavefront sensor cmos
            set_camera_parameters(cam)
            #take frame and plot it
            frame = grab_frame(cam)
            fig = plt.figure()
            gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
            ax1 = plt.subplot((gs[0]))
            image = ax1.imshow(frame)
            cbar1 = fig.colorbar(image, ax=ax1, label='ADUs', fraction=0.046)
            # Plot the grid
            ax1.axvline(155, color='red')
            ax1.axvline(155 + 1, color='red')
            ax1.axhline(155, color='red')
            ax1.axhline(155 + 1, color='red')
            # process frame in gpu
            img = torch.from_numpy(frame).to(device).squeeze()
            subaps = split_wfs_image(img)
            centroids = center_of_gravity(subaps)
            slopes = centroids_to_slopes(centroids)
            slopes_x, slopes_y = hudgin_slopes(slopes)
            phase = torch.linalg.lstsq(B, torch.cat([slopes_x, slopes_y, torch.zeros(1, device=device)])).solution * 500
            #  phase -= phase.mean()
            ax2 = plt.subplot(gs[1])
            r_phase = ax2.imshow(phase.reshape(11, 11).to('cpu'), cmap='plasma')
            cbar2 = fig.colorbar(r_phase, ax=ax2, label='microns', fraction=0.046)
            ax3 = plt.subplot(gs[2])
            slopes_plot, = ax3.plot(torch.cat([slopes_x, slopes_y]).to('cpu'))
            ani = FuncAnimation(fig, partial(update, image=image, r_phase=r_phase, slopes_plot=slopes_plot, cam=cam,
                                              cbar1=cbar1, cbar2=cbar2, B=B), interval=0, cache_frame_data=False)
            plt.show()

def main_still_image():

    print('Building Geometry Matrix\n')
    B = generate_B_matrix(11)
    frame = np.load("wfs-frame.npy")
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
    ax1 = plt.subplot((gs[0]))
    image = ax1.imshow(frame)
    cbar1 = fig.colorbar(image, ax=ax1, label='ADUs', fraction=0.046)
    # Plot the grid
    ax1.axvline(155, color='red')
    ax1.axvline(155 + 1, color='red')
    ax1.axhline(155, color='red')
    ax1.axhline(155 + 1, color='red')
    # process frame in gpu
    img = torch.from_numpy(frame).to(device, dtype=torch.float32).squeeze()
    subaps = split_wfs_image(img)
    centroids = center_of_gravity(subaps)
    slopes = centroids_to_slopes(centroids)
    slopes_x, slopes_y = hudgin_slopes(slopes)
    phase = torch.linalg.lstsq(B, torch.cat([slopes_x, slopes_y, torch.zeros(1, device=device)])).solution * 500
    #  phase -= phase.mean()
    ax2 = plt.subplot(gs[1])
    r_phase = ax2.imshow(phase.reshape(11, 11).to('cpu'), cmap='plasma')
    cbar2 = fig.colorbar(r_phase, ax=ax2, label='microns', fraction=0.046)
    ax3 = plt.subplot(gs[2])
    slopes_plot, = ax3.plot(torch.cat([slopes_x, slopes_y]).to('cpu'))
    plt.show()


if __name__ == "__main__":
    main_video_stream()
    # main_still_image()
