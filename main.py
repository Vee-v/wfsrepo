from wfstools import *
from wfsstartup import startup
from vmbpy import VmbSystem
import queue
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

if __name__ == "__main__":
    print("Starting up the camera...")
    B = generate_B_matrix(11)
    frame_queue = queue.Queue(maxsize=5)  # Buffer for frames between threads

    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        with cams[0] as cam:
            reference_positions, camera_thread, valid_subap_mask = startup(cam)
            slopes_thread = SlopesThread(frame_queue, reference_positions, valid_subap_mask)
            slopes_thread.start()
            try:
                while True:
                    frame = camera_thread.get_frame(timeout=1)
                    if frame is not None:
                        try:
                            frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass  # Drop frame if queue is full
                    cam_fps = camera_thread.get_fps()
                    slopes_fps = slopes_thread.get_fps()
                    latest_slopes = slopes_thread.get_slopes()

                    # Display frame and slopes plot side by side
                    if frame is not None and latest_slopes is not None:
                        # Prepare the frame for display (grayscale to BGR if needed)
                        if len(frame.shape) == 2:
                            frame_disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        else:
                            frame_disp = frame.copy()
                        # Prepare the slopes plot
                        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
                        ax.plot(np.concatenate([latest_slopes[:,0], latest_slopes[:,1]]))
                        ax.set_title('Latest Slopes')
                        ax.set_xlabel('Index')
                        ax.set_ylabel('Slope')
                        fig.tight_layout()
                        canvas = FigureCanvas(fig)
                        canvas.draw()
                        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        plt.close(fig)
                        # Resize plot to match frame height
                        h_frame = frame_disp.shape[0]
                        h_plot, w_plot, _ = plot_img.shape
                        scale = h_frame / h_plot
                        plot_img_resized = cv2.resize(plot_img, (int(w_plot*scale), h_frame))
                        # Concatenate images
                        combined = np.concatenate((frame_disp, plot_img_resized), axis=1)
                        cv2.imshow('Frame and Slopes', combined)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    print(f"Camera FPS: {cam_fps:.2f} | Slopes FPS: {slopes_fps:.2f}", end='\r')
            except KeyboardInterrupt:
                print("\nStopping all threads and exiting...")
            finally:
                camera_thread.stop()
                slopes_thread.stop()
                cv2.destroyAllWindows()