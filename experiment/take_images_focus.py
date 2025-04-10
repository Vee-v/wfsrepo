from misalignment_calib import *
from pathlib import Path

output = Path("measurements")

def take_images(d, t_exp=50):
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            # set parameters for wavefront sensor cmos
            set_camera_parameters(cam, t_exp=t_exp)

            frame = grab_frame(cam).squeeze()
        
            np.save(output / f"focus_img_{d}.npy", frame)
            plt.figure()
            plt.imshow(frame)
            plt.colorbar()
            plt.show() 
    return 

while True:
    take_images(input("Input housing depth in mm:\n"))