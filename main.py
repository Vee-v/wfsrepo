from wfstools import *
from wfsstartup import startup
from vmbpy import VmbSystem
import queue
import cv2
import socket
import numpy as np
import pickle

host = "10.11.42.189"
port = 5000

if __name__ == "__main__":
    print("Starting up the camera...")
    B = generate_B_matrix(11)
    frame_queue = queue.Queue(maxsize=5)  # Buffer for frames between camera and slopes threads

    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            reference_positions, camera_thread, valid_subap_mask = startup(cam)
            slopes_thread = SlopesThread(frame_queue, reference_positions, valid_subap_mask)
            slopes_thread.start()
            # reduce exposure time
            cam.ExposureTime.set(21.481*3)
            # Precompute Zernike design matrix for valid subaps
            input("Press Enter to continue and build imat...")
            N_zernike = int(input("How many zernike modes to plot? \n"))
            # Use subaperture center positions for Zernike fit
            subap_positions = calculate_subaperture_positions(grid_size=11, subap_size=28)  # shape [121, 2], in pixels
            valid_subap_positions = subap_positions[valid_subap_mask].detach().cpu()
            zernike_A = build_zernike_derivative_matrix(valid_subap_positions, N_zernike)

            # Use a one-element list for thread-safe frame sharing
            latest_frame = [None]
            frame_queue_thread = FrameQueueThread(camera_thread, frame_queue, latest_frame)
            frame_queue_thread.start()

            try:
                with socket.socket() as client_socket:
                    client_socket.connect((host, port))  # Connect to the DM server
                    # get number of subaps valid
                    num_valid_subaps = np.sum(valid_subap_mask.detach().cpu().numpy(), dtype=int).item()
                    print(f"Number of valid subapertures: {num_valid_subaps}")
                    # Send the number of slopes to the DM server
                    client_socket.sendall(num_valid_subaps.to_bytes(4, byteorder='big'))
                    # interaction matrix building loop
                    while True:    
                        msg = client_socket.recv(2086).decode()
                        if msg != "OK":
                            print("finished imat")
                            break
                        frame = latest_frame[0]
                        if frame is not None:
                            latest_slopes = slopes_thread.get_slopes()
                            if latest_slopes is not None:
                                # Send slopes to the DM server
                                slopes = latest_slopes.flatten().detach().cpu().numpy()
                                pickled = pickle.dumps(slopes.astype(np.float32))
                                client_socket.send(len(pickled).to_bytes(4, byteorder='big'))
                                client_socket.send(pickled) 
                    while True:
                        frame = latest_frame[0]
                        if frame is not None:
                            if latest_slopes is not None:
                                zernike_coeffs = fit_slopes_to_zernike(latest_slopes, zernike_A)
                                # Make the bar plots as tall as the frame, and wide enough for readability
                                slopes_barplot_img = show_slopes_barplot_opencv(latest_slopes, height=frame.shape[0], frame_height=frame.shape[0])
                                zernike_barplot_img = show_zernike_barplot_opencv(zernike_coeffs, height=frame.shape[0], frame_height=frame.shape[0])
                                # Ensure frame is 3-channel for stacking
                                if frame.ndim == 2:
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                else:
                                    frame_bgr = frame
                                # Horizontally stack frame, slopes barplot, and zernike barplot
                                combined = np.hstack([frame_bgr, slopes_barplot_img, zernike_barplot_img])
                                cv2.imshow('WFS frame', combined)
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q'):
                                    break
                            else:
                                cv2.imshow('WFS frame', frame)
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q'):
                                    break
                        print(f"Frame queue size: {frame_queue.qsize()} ", end=' | ')
                        cam_fps = camera_thread.get_fps()
                        slopes_fps = slopes_thread.get_fps()
                        latest_slopes = slopes_thread.get_slopes()
                        # Calculate and display Zernike coefficients if slopes are available
                        if latest_slopes is not None:
                            # send slopes to the DM server
                            slopes = latest_slopes.flatten().detach().cpu().numpy()
                            pickled = pickle.dumps(slopes.astype(np.float32))
                            client_socket.send(len(pickled).to_bytes(4, byteorder='big'))
                            client_socket.send(pickled) 
                            msg = client_socket.recv(2086).decode()
                            if msg != "OK":
                                print("ERROR: DM server did not respond with OK")
                                break
                        #     zernike_coeffs = fit_slopes_to_zernike(latest_slopes, zernike_A)
                        print(f"Camera FPS: {cam_fps:.2f} | Slopes FPS: {slopes_fps:.2f}", end='\r')
            except KeyboardInterrupt:
                pass
            
            print("\nStopping all threads and exiting...")
            frame_queue_thread.stop()
            camera_thread.stop()
            slopes_thread.stop()
            cv2.destroyAllWindows()