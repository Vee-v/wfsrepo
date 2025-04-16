from wfstools import grab_frames_async
from wfsstartup import startup
from vmbpy import VmbSystem
import time

if __name__ == "__main__":
    print("Starting up the camera...")
    # Open camera and pass it to startup
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            reference_positions, camera_thread, valid_subap_mask = startup(cam)
            # Now that we have the camera thread, the subaperture mask and the reference centroid positions,
            # we start wavefront sensing.
            try:
                while True:
                    frame = camera_thread.get_frame(timeout=1)
                    fps = camera_thread.get_fps()
                    print(f"Current FPS: {fps}", end='\r')
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping all threads and exiting...")
                camera_thread.stop()