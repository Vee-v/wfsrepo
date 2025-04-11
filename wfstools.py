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

mla_intr_shift= np.load(Path("experiments") / "delta-centroid-empirical.npy")


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
    def frame_handler(cam, frame):
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