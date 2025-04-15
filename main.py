from wfstools import grab_frames_async
from wfsstartup import startup
from vmbpy import VmbSystem

if __name__ == "__main__":

    print("Starting up the camera...")
    # Open camera and pass it to startup
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            reference_positions, camera_thread = startup(cam)
            try:
                grab_frames_async(camera_thread=camera_thread)
            finally:
                camera_thread.stop()