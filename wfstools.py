import torch
import matplotlib.pyplot as plt
import numpy as np
import threading
import queue
import time
from torchvision.transforms import v2
from threading import Event
from pathlib import Path
from functools import partial
from vmbpy import VmbSystem
from vmbpy.camera import Camera
from vmbpy import PixelFormat
from scipy.optimize import curve_fit
from tqdm import tqdm
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mla_intr_shift= np.load(Path("experiment") / "delta-centroid-empirical.npy") * 1000 / 18  # from mm to pixels


class CameraThread:
    def __init__(self, cam):
        self.cam = cam
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = threading.Event()
        self.thread = threading.Thread(target=self._run)
        self.fps = 0.0
        self._frame_count = 0
        self._fps_lock = threading.Lock()
        self._watchdog_thread = threading.Thread(target=self._watchdog)
        self._new_frame_event = threading.Event()

    def start(self):
        self.running.set()
        self.thread.start()
        self._watchdog_thread.start()

    def stop(self):
        self.running.clear()
        self.thread.join()
        self._watchdog_thread.join()

    def _frame_handler(self, cam, stream, frame):
        img = frame.as_numpy_ndarray().squeeze()
        with self.frame_lock:
            self.latest_frame = img
        with self._fps_lock:
            self._frame_count += 1
        self._new_frame_event.set()
        cam.queue_frame(frame)

    def _run(self):
        self.cam.start_streaming(self._frame_handler)
        while self.running.is_set():
            threading.Event().wait(0.01)  # Sleep briefly to yield thread
        self.cam.stop_streaming()

    def _watchdog(self):
        while self.running.is_set():
            with self._fps_lock:
                start_count = self._frame_count
            time.sleep(1.0)
            with self._fps_lock:
                end_count = self._frame_count
                self._frame_count = 0  # Reset frame count after getting FPS
            self.fps = end_count - start_count

    def get_frame(self, timeout=1):
        if self._new_frame_event.wait(timeout=timeout):
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            self._new_frame_event.clear()
            return frame
        return None

    def get_fps(self):
        return self.fps


class PhaseThread:
    def __init__(self, frame_queue, B, reference_positions, valid_subap_mask, display_phase=False):
        self.frame_queue = frame_queue
        self.B = B
        self.reference_positions = reference_positions
        self.valid_subap_mask = valid_subap_mask
        self.latest_phase = None
        self.phase_lock = threading.Lock()
        self.running = threading.Event()
        self.thread = threading.Thread(target=self._run)
        self.fps = 0.0
        self._frame_count = 0
        self._fps_lock = threading.Lock()
        self._watchdog_thread = threading.Thread(target=self._watchdog)
        self.display_phase = display_phase

    def start(self):
        self.running.set()
        self.thread.start()
        self._watchdog_thread.start()

    def stop(self):
        self.running.clear()
        self.thread.join()
        self._watchdog_thread.join()
        if self.display_phase:
            cv2.destroyAllWindows()

    def _run(self):
        while self.running.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            img = torch.from_numpy(frame).to(device, dtype=torch.float32).squeeze()
            subaps = split_wfs_image(img)
            centroids = center_of_gravity(subaps)
            slopes = centroids_to_slopes(centroids, self.reference_positions)
            slopes_x, slopes_y = hudgin_slopes(slopes)
            phase = (torch.linalg.lstsq(self.B, torch.cat([slopes_x, slopes_y, torch.zeros(1, device=device)])).solution / (2*np.pi)) * 633e-3
            with self.phase_lock:
                self.latest_phase = phase
            with self._fps_lock:
                self._frame_count += 1
            if self.display_phase:
                # Normalize phase for display
                phase_np = phase.detach().cpu().numpy()
                phase_img = phase_np[:-1].reshape(11, 11) if phase_np.shape[0] == 122 else phase_np.reshape(11, 11)
                phase_img_resized = cv2.resize(phase_img, (308, 308), interpolation=cv2.INTER_CUBIC)
                norm_phase = cv2.normalize(phase_img_resized, None, 0, 255, cv2.NORM_MINMAX)
                norm_phase = norm_phase.astype(np.uint8)
                # Prepare frame for display (normalize and convert to uint8)
                norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                norm_frame = norm_frame.astype(np.uint8)
                frame_resized = cv2.resize(norm_frame, (308, 308), interpolation=cv2.INTER_CUBIC)
                combined = np.hstack((frame_resized, norm_phase))
                cv2.imshow('Frame (left) | Phase (right)', combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running.clear()
                    break

    def _watchdog(self):
        while self.running.is_set():
            with self._fps_lock:
                start_count = self._frame_count
            time.sleep(1.0)
            with self._fps_lock:
                end_count = self._frame_count
                self._frame_count = 0
            self.fps = end_count - start_count

    def get_phase(self):
        with self.phase_lock:
            return self.latest_phase

    def get_fps(self):
        return self.fps


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

def set_camera_parameters(camera: Camera, t_exp=None):
    with camera:
        ''' 
        According to the manual, the order of parameter setting needs to be the following (higher parameters affect lower parameters):
            1. Pixel format (MONO8)
            2. Sensor bit depth (MONO8)
            3. Device throughput limit (450MB/s)
            4. Black level (0)
            5. Reverse X/Y (not reverse)
            6. Binning (2x2)
            7. ROI settings (not sure yet)
            8. Exposure time and framerate (619fps should be fastest?)
        '''
        # Set Pixel format and bit depth
        camera.set_pixel_format(PixelFormat.Mono8)
        camera.SensorBitDepth.set("Bpp8")  # Set sensor bit depth to 8 bits
        # Set device throughput limit
        camera.DeviceLinkThroughputLimit.set(450000000)
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
            camera.ExposureTime.set(1200)  # in microseconds
        else:
            camera.ExposureTime.set(t_exp)

        
        # Set gamma
        camera.Gamma.set(1.)

        # Set gain
        camera.Gain.set(1)

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

def centroids_to_slopes(centroids, reference):
    slopes = (centroids - reference) * 18 / 13800  # radian
    return torch.arcsin(slopes)

def hudgin_slopes(slopes):  # optimizar
    edge_slopes = list(range(10, 121, 11))
    slopes_x = slopes[:, 0]
    slopes_y = slopes[:-11, 1]
    mask = torch.ones(slopes.size(0), dtype=torch.bool)
    mask[edge_slopes] = False
    slopes_x = slopes_x[mask]
    return slopes_x, slopes_y

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

    return np.arctan(popt[0])

def calculate_reference(subap_positions, theta, deltas=torch.zeros(1, dtype=torch.float32)):
    """
    Calculates correct reference by comparing centroids to rotated expected positions.

    Args:
        subap_positions (Tensor): Subaperture positions in the Shack-Hartmann grid (N^2, 2) in pixels.
        theta (float): Rotation angle of the lenslet array in radians.
        deltas (Tensor): Array of tip-tilt offsets in pixels to be added to the reference centroids.

    Returns:
        reference_centroids (Tensor): Reference centroid position for flat wavefront in pixels.
    """
    # Rotate subaperture positions
    cos_theta, sin_theta = np.cos(-theta), np.sin(-theta)
    rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], device=device, dtype=torch.float32)
    rotated_positions = (subap_positions @ rotation_matrix.T)
    # Calculate the expected centroids after accounting for rotation
    reference_centroids = rotated_positions - subap_positions + deltas.to(device)/18. + 13.5  # pixels

    return reference_centroids

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

def get_valid_subaps_mask(subaps, noise_baseline, factor=3, min_pixels=3):
    """
    Returns a boolean mask indicating which subapertures are valid.
    A subaperture is valid if it contains at least min_pixels pixels above factor * noise_baseline.
    
    Args:
        subaps (Tensor): (N, 28, 28) tensor of subaperture images.
        noise_baseline (float or Tensor): The noise baseline value.
        factor (float): The factor above the baseline to consider a pixel "active".
        min_pixels (int): Minimum number of active pixels for a subaperture to be valid.
    
    Returns:
        Tensor: Boolean mask of shape (N,) indicating valid subapertures.
    """
    threshold = factor * noise_baseline
    active_pixels = (subaps > threshold).sum(dim=(1,2))
    return active_pixels >= min_pixels


def grab_frames_async(camera_thread=None):
    """
    Displays frames in real time from a CameraThread buffer if provided, else starts camera stream.
    """
    if camera_thread is not None:
        print("Displaying Frames in Real Time from CameraThread! Press 'q' to quit.")
        import cv2
        while True:
            frame = camera_thread.get_frame(timeout=2)
            if frame is not None:
                cv2.imshow('Live Frame', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        cv2.destroyAllWindows()
        return

    def frame_handler(cam, stream, frame):  
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

def grab_frames_to_array(cam, n_frames, camera_thread=None):
    """
    Grabs a specified number of frames from the camera or from a CameraThread buffer.

    Args:
        cam (Camera): The camera object to grab frames from.
        n_frames (int): Number of frames to grab.
        camera_thread (CameraThread, optional): If provided, use this thread's buffer.

    Returns:
        np.ndarray: Array of grabbed frames with shape (n_frames, height, width).
    """
    frames = []
    if camera_thread is not None:
        for _ in tqdm(range(n_frames), desc="Grabbing frames (thread)", unit="frame"):
            frame = camera_thread.get_frame()
            if frame is not None:
                frames.append(frame)
        return np.array(frames)

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