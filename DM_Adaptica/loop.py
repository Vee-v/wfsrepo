import time
import socket
import pickle
import numpy as np
import matplotlib.pyplot as plt
from saturn import deformableMirror


host = "10.11.42.189"
port = 5000
num_slopes = 242
reps = 1  #number of movement repetition for imat building

# initiliaze de DM
saturn = deformableMirror()
saturn.reset_all_channels()

aux = np.zeros((num_slopes, reps), dtype=np.float32)
interaction_matrix = np.zeros((num_slopes, saturn.num_channels), dtype=np.float32)



# initilize socket server to listen for slope arrays
with socket.socket() as server_socket:
    server_socket.bind((host, port))
    try:
        server_socket.listen()
        conn, address = server_socket.accept()
        # build interaction matrix (imat)
        for ch in range(saturn.num_channels):
            saturn.set_channels(np.ones(saturn.num_channels, dtype=np.float32) * 0.5)
            for rep in range(reps):
                # push
                saturn.set_single_channel(1.0, ch)
                time.sleep(0.1)
                conn.send("OK".encode())
                slopes = conn.recv(2086)
                try:
                    slopes = pickle.loads(slopes)
                    aux[:, rep] += slopes
                except pickle.UnpicklingError as e:
                    print("Error in unpickling:", e)
                # pull
                saturn.set_single_channel(0., ch)
                time.sleep(0.1)
                conn.send("OK".encode())
                slopes = conn.recv(2086)
                try:
                    slopes = pickle.loads(slopes)
                    aux[:, rep] -= slopes
                except pickle.UnpicklingError as e:
                    print("Error in unpickling:", e)
            interaction_matrix[:, ch] = np.nanmean(aux/reps, axis=1)
        plt.matshow(interaction_matrix, aspect="auto")
        plt.colorbar()
        plt.show()
        saturn.disconnect_mirror()
        exit(0)
        conn.send("finished".encode())
        while True:
            slopes = conn.recv(2086)
            time.sleep(1e-9)
            try:
                pickle.loads(slopes)
            except pickle.UnpicklingError as e:
                print("Error in unpickling:", e)
    except KeyboardInterrupt:
        print("\nConnection interrupted by user.")

saturn.disconnect_mirror()
