import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from threading import Event
from pathlib import Path
from functools import partial
from vmbpy import VmbSystem
from vmbpy.camera import Camera
from vmbpy import PixelFormat
from scipy.optimize import curve_fit
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mla_intr_shift= np.load(Path("experiment") / "delta-centroid-empirical.npy")


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
        camera.OffsetX.set(0)  # 816/2=408; 408-304=104; 104/2=52; for some reason the offset in X is now at 0.
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

def center_of_gravity_np(img):
 
    y_coords, x_coords = np.repeat(np.arange(28), 28).reshape(28, 28), np.transpose(np.repeat(np.arange(28), 28).reshape(28, 28))
    total_weight = np.sum(img)
    x_centroid = np.sum((img * x_coords)) / total_weight
    y_centroid = np.sum((img * y_coords)) / total_weight

    return [x_centroid, y_centroid]

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

def take_images(n=100, t_exp=100):
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            # set parameters for wavefront sensor cmos
            set_camera_parameters(cam, t_exp=t_exp)

            frames = np.zeros((n, 312, 312))
            # frames = np.expand_dims(frames, axis=0)
            print("Taking Images!\n")
            for i in range(n):
                frames[i, :, :] = grab_frame(cam).squeeze()
            frame = np.mean(frames, axis=0).squeeze()
            np.save("img.npy", frame)
            plt.figure()
            plt.imshow(frame)
            plt.colorbar()
            plt.show() 
    return

def grab_frames_async():
    def frame_handler(cam, stream, frame):  # Added 'stream' as the second argument
        nonlocal stop_event, exposure_time
        img = frame.as_numpy_ndarray().squeeze()
        cv2.imshow('Live Frame', img)

        # Get FPS from the camera if supported
        try:
            fps = cam.AcquisitionFrameRate.get()
        except AttributeError:
            fps = "N/A"  # If the feature is not supported, fallback to "N/A"

        # Update window title with FPS and exposure time
        cv2.setWindowTitle('Live Frame', f'FPS: {fps}, Exposure: {exposure_time} µs')

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
        elif key == ord('+'):
            try:
                exposure_time += 10
                cam.ExposureTime.set(exposure_time)
                print(f"Increased exposure time to {exposure_time} µs")
            except Exception as e:
                print(f"Failed to increase exposure time: {e}")
        elif key == ord('-'):
            try:
                exposure_time = max(10, exposure_time - 10)
                cam.ExposureTime.set(exposure_time)
                print(f"Decreased exposure time to {exposure_time} µs")
            except Exception as e:
                print(f"Failed to decrease exposure time: {e}")
        cam.queue_frame(frame)

    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            set_camera_parameters(cam)

            stop_event = Event()
            exposure_time = cam.ExposureTime.get()  # Get initial exposure time

            cam.start_streaming(frame_handler)
            print("Displaying Frames in Real Time! Press 'q' to quit, '+' to increase exposure, '-' to decrease exposure.\n")

            stop_event.wait()  # Wait until 'q' is pressed
            cam.stop_streaming()
            cv2.destroyAllWindows()

    return

def grab_frames_to_array(cam, n_frames):
    """
    Grabs a specified number of frames from the camera and saves them into an array.

    Args:
        cam (Camera): The camera object to grab frames from.
        n_frames (int): Number of frames to grab.

    Returns:
        np.ndarray: Array of grabbed frames with shape (n_frames, height, width).
    """
    frames = []

    def frame_handler(cam, stream, frame):
        nonlocal frames, progress_bar
        img = frame.as_numpy_ndarray().squeeze()
        frames.append(img)
        progress_bar.update(1)  # Update the progress bar
        cam.queue_frame(frame)

        # Stop streaming once the required number of frames is collected
        if len(frames) >= n_frames:
            stop_event.set()

    stop_event = Event()
    progress_bar = tqdm(total=n_frames, desc="Grabbing frames", unit="frame")

    cam.start_streaming(frame_handler)

    stop_event.wait()  # Wait until the required number of frames is collected

    cam.stop_streaming()
    progress_bar.close()  # Close the progress bar

    return np.array(frames)