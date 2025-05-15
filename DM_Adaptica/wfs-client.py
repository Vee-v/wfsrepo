import numpy as np
import pickle
import socket
import time


host = "10.11.42.189"
port = 5000
slopes = np.arange(242, dtype=float)
print(f" size in bytes: {len(pickle.dumps(slopes))}")
with socket.socket() as client_socket:
    client_socket.connect((host, port))
    while True:    
        msg = client_socket.recv(2086).decode()
        if msg != "OK":
            print("finished imat")
            break
        client_socket.sendall(pickle.dumps(slopes))      
    while True:
        client_socket.sendall(pickle.dumps(slopes))
        
