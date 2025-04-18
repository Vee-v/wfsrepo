from wfstools import *
from wfsstartup import startup
from vmbpy import VmbSystem
import queue
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from threading import Event

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

            # Use a one-element list for thread-safe frame sharing
            latest_frame = [None]
            frame_queue_thread = FrameQueueThread(camera_thread, frame_queue, latest_frame)
            frame_queue_thread.start()

            try:
                while True:
                    frame = latest_frame[0]
                    if frame is not None:
                        cv2.imshow('WFS frame', frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    print(f"Frame queue size: {frame_queue.qsize()} ", end=' | ')
                    cam_fps = camera_thread.get_fps()
                    slopes_fps = slopes_thread.get_fps()
                    latest_slopes = slopes_thread.get_slopes()
                    print(f"Camera FPS: {cam_fps:.2f} | Slopes FPS: {slopes_fps:.2f}", end='\r')
            except KeyboardInterrupt:
                print("\nStopping all threads and exiting...")
            finally:
                frame_queue_thread.stop()
                camera_thread.stop()
                slopes_thread.stop()
                cv2.destroyAllWindows()