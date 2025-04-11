from wfstools import grab_frames_async, set_camera_parameters
from vmbpy import VmbSystem

if __name__ == "__main__":
    # Example usage of the grab_frames_async function
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            # set parameters for wavefront sensor cmos
            set_camera_parameters(cam, t_exp=100)
    grab_frames_async()