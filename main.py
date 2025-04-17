from wfstools import *
from wfsstartup import startup
from vmbpy import VmbSystem
import queue
import cv2
import numpy as np

if __name__ == "__main__":
    print("Starting up the camera...")
    B = generate_B_matrix(11)
    frame_queue = queue.Queue(maxsize=5)  # Buffer for frames between threads

    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            reference_positions, camera_thread, valid_subap_mask = startup(cam)
            phase_thread = PhaseThread(frame_queue, B, reference_positions, valid_subap_mask)
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
                    latest_phase = phase_thread.get_phase()
                    if latest_phase is not None and frame is not None:
                        # Normalize phase for display
                        phase_np = latest_phase.detach().cpu().numpy()
                        phase_img = phase_np[:-1].reshape(11, 11) if phase_np.shape[0] == 122 else phase_np.reshape(11, 11)
                        # Interpolate phase to 308x308
                        phase_img_resized = cv2.resize(phase_img, (308, 308), interpolation=cv2.INTER_CUBIC)
                        norm_phase = cv2.normalize(phase_img_resized, None, 0, 255, cv2.NORM_MINMAX)
                        norm_phase = norm_phase.astype(np.uint8)
                        # Prepare frame for display (normalize and convert to uint8)
                        norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                        norm_frame = norm_frame.astype(np.uint8)
                        # Ensure frame is 2D, resize to 308x308
                        frame_resized = cv2.resize(norm_frame, (308, 308), interpolation=cv2.INTER_CUBIC)
                        # Stack phase and frame side by side
                        combined = np.hstack((frame_resized, norm_phase))
                        cv2.imshow('Frame (left) | Phase (right)', combined)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    print(f"Camera FPS: {cam_fps:.2f} | Phase FPS: {phase_fps:.2f}", end='\r')
            except KeyboardInterrupt:
                print("\nStopping all threads and exiting...")
            finally:
                camera_thread.stop()
                phase_thread.stop()
                cv2.destroyAllWindows()