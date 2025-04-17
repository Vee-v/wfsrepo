from wfstools import *
from wfsstartup import startup
from vmbpy import VmbSystem
import queue
import cv2

if __name__ == "__main__":
    print("Starting up the camera...")
    B = generate_B_matrix(11)
    frame_queue = queue.Queue(maxsize=5)  # Buffer for frames between threads

    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            reference_positions, camera_thread, valid_subap_mask = startup(cam)
            # Enable phase display in PhaseThread
            phase_thread = PhaseThread(frame_queue, B, reference_positions, valid_subap_mask, display_phase=True)
            phase_thread.start()
            try:
                while True:
                    frame = camera_thread.get_frame(timeout=1)
                    if frame is not None:
                        try:
                            frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass  # Drop frame if queue is full
                    cam_fps = camera_thread.get_fps()
                    phase_fps = phase_thread.get_fps()
                    # No more phase display here
                    print(f"Camera FPS: {cam_fps:.2f} | Phase FPS: {phase_fps:.2f}", end='\r')
            except KeyboardInterrupt:
                print("\nStopping all threads and exiting...")
            finally:
                camera_thread.stop()
                phase_thread.stop()
                cv2.destroyAllWindows()