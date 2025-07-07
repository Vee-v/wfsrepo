from wfstools import *
from wfsstartup import startup
from vmbpy import VmbSystem
import queue
import cv2
import numpy as np

def overlay_subaperture_hues(frame, valid_subap_mask, grid_size=11, subap_size=28, alpha=0.1):
    """
    Overlay green on valid subaps and red on invalid subaps.
    frame: (H, W) or (H, W, 3) numpy array
    valid_subap_mask: (N,) boolean array
    """
    if frame.ndim == 2:
        frame_bgr = cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_GRAY2BGR)
    else:
        frame_bgr = frame.copy()
    overlay = frame_bgr.copy()
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * subap_size, (i + 1) * subap_size
            x0, x1 = j * subap_size, (j + 1) * subap_size
            color = (0, 255, 0) if valid_subap_mask[idx] else (0, 0, 255)  # Green or Red (BGR)
            cv2.rectangle(overlay, (x0, y0), (x1-1, y1-1), color, thickness=-1)
            idx += 1
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)
    return frame_bgr

if __name__ == "__main__":
    print("Starting up the camera...")
    B = generate_B_matrix(11)
    frame_queue = queue.Queue(maxsize=5)  # Buffer for frames between camera and slopes threads

    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            reference_positions, camera_thread, valid_subap_mask = startup(cam)
            slopes_thread = SlopesThread(frame_queue, reference_positions, valid_subap_mask, angular=True)
            slopes_thread.start()
            # reduce exposure time
            cam.ExposureTime.set(33.324)  # 33.324 ms for red laser
            # Precompute Zernike design matrix for valid subaps
            input("Remove calibration lamp... \nPress Enter to continue and compute Zernike coefficients...")
            N_zernike = int(input("How many zernike modes to plot? \n"))
            # Use subaperture center positions for Zernike fit
            subap_positions = calculate_subaperture_positions(grid_size=11, subap_size=28)  # shape [121, 2], in pixels
            valid_subap_positions = subap_positions[valid_subap_mask].detach().cpu()
            zernike_A = build_zernike_derivative_matrix(valid_subap_positions, N_zernike)

            # Use a one-element list for thread-safe frame sharing
            latest_frame = [None]
            frame_queue_thread = FrameQueueThread(camera_thread, frame_queue, latest_frame)
            frame_queue_thread.start()
            latest_slopes = None
            try:
                while True:
                    frame = latest_frame[0]
                    if frame is not None:
                        # Overlay hues on WFS frame
                        frame_with_hues = overlay_subaperture_hues(frame, valid_subap_mask)
                        if latest_slopes is not None:
                            zernike_coeffs = fit_slopes_to_zernike(latest_slopes, zernike_A)
                            # Make the bar plots as tall as the frame, and wide enough for readability
                            slopes_barplot_img = show_slopes_barplot_opencv(latest_slopes, height=frame.shape[0], frame_height=frame.shape[0])
                            zernike_barplot_img = show_zernike_barplot_opencv(zernike_coeffs, height=frame.shape[0], frame_height=frame.shape[0])
                            if frame_with_hues.ndim == 2:
                                frame_bgr = cv2.cvtColor(frame_with_hues.astype("uint8"), cv2.COLOR_GRAY2BGR)
                            else:
                                frame_bgr = frame_with_hues
                            # Horizontally stack frame, slopes barplot, and zernike barplot
                            combined = np.hstack([frame_bgr, slopes_barplot_img, zernike_barplot_img])
                            cv2.imshow('WFS frame', combined)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                break
                        else:
                            cv2.imshow('WFS frame', frame_with_hues)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                break
                    print(f"Frame queue size: {frame_queue.qsize()} ", end=' | ')
                    cam_fps = camera_thread.get_fps()
                    slopes_fps = slopes_thread.get_fps()
                    latest_slopes = slopes_thread.get_slopes()
                    # Calculate and display Zernike coefficients if slopes are available
                    if latest_slopes is not None:
                        zernike_coeffs = fit_slopes_to_zernike(latest_slopes, zernike_A)
                    print(f"Camera FPS: {cam_fps:.2f} | Slopes FPS: {slopes_fps:.2f}", end='\r')
            except KeyboardInterrupt:
                print("\nStopping all threads and exiting...")
            finally:
                frame_queue_thread.stop()
                camera_thread.stop()
                slopes_thread.stop()
                cv2.destroyAllWindows()