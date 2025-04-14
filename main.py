from wfstools import grab_frames_async, set_camera_parameters
from wfsstartup import startup
from vmbpy import VmbSystem

if __name__ == "__main__":

    print("Starting up the camera...")
    # Example usage of the startup function to get reference positions
    reference_positions = startup()
    input("Press Enter to start the camera and grab frames.")
    # Example usage of the grab_frames_async function
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            # set parameters for wavefront sensor cmos
            set_camera_parameters(cam, t_exp=100)
    grab_frames_async()