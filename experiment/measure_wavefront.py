from misalignment_calib import *
from pathlib import Path

output = Path('measurements')
theta = np.load("theta.npy")


def take_images(d, theta, deltas, n=1000, t_exp=1000):
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            # set parameters for wavefront sensor cmos
            set_camera_parameters(cam, t_exp=t_exp)
            reference_positions = calculate_reference(subap_positions, theta, deltas)

            frames = np.zeros((n, 312, 312))
            # frames = np.expand_dims(frames, axis=0)
            print("Taking Images!\n")
            for i in tqdm(range(n)):
                frames[i, :, :] = grab_frame(cam).squeeze()
            frame = np.mean(frames, axis=0).squeeze()
            img = torch.from_numpy(frame).to(device, dtype=torch.float32).squeeze()
            subaps = split_wfs_image(img)
            centroids = center_of_gravity(subaps)
            slopes = centroids_to_slopes(centroids, reference_positions)  # radians
            slopes_x, slopes_y = hudgin_slopes(slopes)
            phase = torch.linalg.lstsq(B, torch.cat([slopes_x, slopes_y, torch.zeros(1, device=device)])).solution / (2*np.pi) * 633e-3  # radians to micrometers
            np.save(output / f"slopes_{d:.3f}.npy", slopes.to('cpu').numpy())
            np.save(output / f"phase_{d:.3f}.npy", phase.to('cpu').numpy())
            np.save(output / f"img_{d:.3f}.npy", frame)
            plt.figure()
            plt.imshow(frame)
            plt.colorbar()
            plt.show() 
    return 

def main():
    d = float(input("Input the distance between the source and the lenslet array in millimeters\n"))
    deltas = np.load(f"deltas_{d:.3f}.npy")
    take_images(d,
                theta=theta,
                deltas = torch.tensor(deltas),
                n=int(input("Input number of images\n")),
                t_exp=int(input("Input exposure time in microseconds\n")))
    
if __name__ == "__main__":
    main()
