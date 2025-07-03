import numpy as np
import cv2
import torch
from wfstools import set_camera_parameters, CameraThread
from vmbpy import VmbSystem
import time

# Initial exposure time (from wfsstartup.py)
exp_time = 21.481

def find_best_exposure_time(cam, camera_thread, max_val=255*.75, min_exp=21.481, max_exp=1000.0, step=1.05, max_iter=20):
    """
    Finds the best exposure time such that no pixels are saturated.
    Args:
        cam: Camera object
        camera_thread: CameraThread object
        max_val: Maximum allowed pixel value (saturation threshold)
        min_exp: Minimum exposure time (ms)
        max_exp: Maximum exposure time (ms)
        step: Multiplicative step for exposure adjustment
        max_iter: Maximum number of iterations
    Returns:
        best_exp: Best exposure time found (ms)
    """
    # Start from current exposure
    exp = cam.ExposureTime.get()
    best_exp = exp
    # First, decrease exposure if saturated
    for _ in range(max_iter):
        cam.ExposureTime.set(exp)
        time.sleep(0.1)
        frame = camera_thread.get_frame(timeout=1)
        if frame is None:
            continue
        max_pixel = np.max(frame)
        if max_pixel >= max_val:
            exp = max(exp / step**2, min_exp)
            print(f"Decreasing exposure time to {exp:.3f} ms (max pixel value: {max_pixel})")
        else:
            break
    # Now, increase exposure until just before saturation
    for _ in range(max_iter):
        cam.ExposureTime.set(exp)
        time.sleep(0.1)
        frame = camera_thread.get_frame(timeout=1)
        if frame is None:
            continue
        max_pixel = np.max(frame)
        if max_pixel >= max_val:
            exp = max(exp / step, min_exp)
            break
        best_exp = exp
        exp = min(exp * step, max_exp)
    cam.ExposureTime.set(best_exp)
    print(f"Best exposure time found: {best_exp:.3f} ms (max pixel value: {max_pixel})")
    return best_exp

def main():
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            print("No cameras found.")
            return
        with cams[0] as cam:
            set_camera_parameters(cam, t_exp=exp_time)
            camera_thread = CameraThread(cam)
            camera_thread.start()
            print("Press 'e' in the image window to change exposure time.")
            print("Press 'a' in the image window to auto-set exposure time.")
            print("Press 'q' in the image window to quit.")
            try:
                while True:
                    frame = camera_thread.get_frame(timeout=1)
                    if frame is not None:

                        frame_disp = frame.copy()
                        # if frame_disp.dtype != np.uint8:
                        #     # Normalize for display
                        #     frame_disp = cv2.normalize(frame_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        cv2.imshow('Camera Frame', frame_disp)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('e'):
                            cv2.destroyWindow('Camera Frame')
                            new_exp = input(f"Current exposure time: {cam.ExposureTime.get():.3f} ms\nEnter new exposure time (ms): ")
                            try:
                                new_exp = float(new_exp)
                                cam.ExposureTime.set(new_exp)
                                print(f"Exposure time set to {cam.ExposureTime.get()} ms.")
                            except Exception as ex:
                                print(f"Invalid input or error: {ex}")
                            # Reopen window
                            cv2.imshow('Camera Frame', frame_disp)
                        elif key == ord('a'):
                            print("Finding best exposure time to avoid saturation...")
                            best_exp = find_best_exposure_time(cam, camera_thread)
                            print(f"Auto exposure set to {best_exp:.3f} ms.")
                    else:
                        time.sleep(0.01)
            finally:
                camera_thread.stop()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
