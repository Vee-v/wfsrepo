import sys
import numpy as np
from ctypes import cdll, c_int, c_void_p, c_float, c_char_p


# C wrapper library:
lib = cdll.LoadLibrary("libadapticaWrapper.so")

lib.Mirror_Connect.argtypes = [c_void_p]
lib.Mirror_Connect.restype = c_int

lib.Mirror_Disconnect.argtypes = [c_void_p]
lib.Mirror_Disconnect.restype = c_int

lib.Mirror_IsConnected.argtypes = [c_void_p]
lib.Mirror_IsConnected.restype = c_int

lib.Mirror_SetChannels.argtypes = [c_void_p, np.ctypeslib.ndpointer(dtype=c_float, ndim=1)]
lib.Mirror_SetChannels.restype = c_int

lib.Mirror_SetSingleChannel.argtypes = [c_void_p, c_float, c_int]
lib.Mirror_SetSingleChannel.restype = c_int

lib.Mirror_GetServerIPAddress.argtypes = [c_void_p]
lib.Mirror_GetServerIPAddress.restype = c_char_p

lib.Mirror_ResetAllChannels.argtypes = [c_void_p]
lib.Mirror_ResetAllChannels.restype = c_int

lib.Mirror_ResetChannel.argtypes = [c_void_p, c_int]
lib.Mirror_ResetChannel.restype = c_int

lib.Mirror_GetChannelsStatus.argtypes = [c_void_p, np.ctypeslib.ndpointer(dtype=c_float, ndim=1)]
lib.Mirror_GetChannelsStatus.restype = c_int

lib.Mirror_GetNumChannels.argtypes = [c_void_p]
lib.Mirror_GetNumChannels.restype = c_int

lib.Mirror_GetSingleChannelStatus.argtypes = [c_void_p, c_int]
lib.Mirror_GetSingleChannelStatus.restype = c_float

lib.Mirror_GetDriverVersion.argtypes = [c_void_p]
lib.Mirror_GetDriverVersion.restype = c_float

lib.Mirror_ConfigServerIPAddress.argtypes = [c_void_p, c_char_p]
lib.Mirror_ConfigServerIPAddress.restype = c_int

lib.Mirror_ResetMirrorDriver.argtypes = [c_void_p]
lib.Mirror_ResetMirrorDriver.restype = c_int

lib.Mirror_GetLastErrorMessage.argtypes = [c_char_p]
lib.Mirror_GetLastErrorMessage.restype = c_char_p


class deformableMirror():
    def __init__(self):
        self.handle = lib.Mirror_Create()
        self.ipaddress = ""
        self.isConnected = 0
        self.num_channels = 0
        self.driver_version = float(0)
        self.connect_mirror()
        if self.isConnected == 0:
            self.config_ipaddress(input("\nPlease enter a valid mirror IP address: "))
        elif self.isConnected != 1:
            print("\nCould not initialize mirror object. Connection error.\n")
            sys.exit(1)
        else:
            self.get_num_channels()
            self.get_driver_version()

        self.channel_voltages = np.ones(self.num_channels, dtype=float)

    def mirror_isconnected(self):
        self.isConnected = lib.Mirror_IsConnected(self.handle)

    def connect_mirror(self):
        self.isConnected = lib.Mirror_Connect(self.handle)
        if self.isConnected != 1:
            print(f"\nCould nonp.ones(1, dtype=c_float)[0] * t connect to mirror. Status: {self.isConnected}\n")
        else:
            print("\nSuccesfully connected to mirror.\n")

    def disconnect_mirror(self):
        status = lib.Mirror_Disconnect(self.handle)
        self.mirror_isconnected()
        if status != 1:
            print("\nCould not disconnect from mirror. Exiting...\n")
            sys.exit(1)
        else:
            print("\nSuccesfully disconnected from mirror\n")
        
    def config_ipaddress(self, ipaddress):
        status = lib.Mirror_ConfigServerIPAddress(self.handle, ipaddress)
        if status != 1:
            print(f"\nCould not configure IP address to: {ipaddress}. Exiting...\n")
            sys.exit(1)
        else:
            self.ipaddress = ipaddress
            print(f"\nSuccesfully configured IP address to: {ipaddress}\n")
    
    def get_num_channels(self):
        self.num_channels = lib.Mirror_GetNumChannels(self.handle)
        print(f"\nNumber of avaiable channels: {self.num_channels}\n")

    def get_driver_version(self):
        self.driver_version = lib.Mirror_GetDriverVersion(self.handle)
        print(f"\nDriver version is: {self.driver_version}\n")
    
    def reset_all_channels(self):
        status = lib.Mirror_ResetAllChannels(self.handle)
        if status != 1:
            print("Could not reset channel voltages.")
        elif status:
            print("Succesfully reset channel voltages.")
        else:
            print(f"Unknown error of status {status}")
    
    def reset_channel(self, ch_id):
        status = lib.Mirror_ResetChannel(self.handle, ch_id)
        if status != 1:
            print(f"Could not reset channel {ch_id} voltage.")
        elif status:
            print(f"Succesfully reset channel {ch_id} voltage.")
        else:
            print(f"Unknown error of status {status}")

    def set_channels(self, data): 
        status = lib.Mirror_SetChannels(self.handle, data)
        if status != 1:
            print("Failed setting channel voltages.")
        elif status:
            self.channel_voltages = data
        else:
            print(f"Unknown error of status {status}")

    def set_single_channel(self, voltage, ch_id):
        c_voltage = c_float(voltage)
        status = lib.Mirror_SetSingleChannel(self.handle, c_voltage, ch_id)
        if status != 1:
            print(f"Failed setting channel {ch_id} voltage to {voltage}V.")
            print(f"Unknown error of status {status}")
        elif status:
            self.channel_voltages[ch_id] = voltage
 

    def get_channels(self):
        voltages = np.ones(self.num_channels, dtype=c_float)
        status = lib.Mirror_GetChannelsStatus(self.handle, voltages)
        # print(voltages.ctypes.data_as(POINTER(c_float)).contents
        if status != 1:
            print(f"Failed getting all channel statuses.")
            print(f"Unknown error of status {status}")
        elif status:
            print(f"All channel voltages: {voltages}")
            
    
    def get_single_channel(self, ch_id):
        status = lib.Mirror_GetSingleChannelStatus(self.handle, ch_id)
        
        print(f"Channel {ch_id} voltage: {status}")

if __name__ == "__main__":
    # This is is a testing script for Adaptica Saturn deformable mirror.
    saturn = deformableMirror()
    print("\nInitialization complete.\n")
    # let's test all channels at once, set a 0, then 0.5 and then 1, repeat.
    input("Press ENTER to test all channels at once\n")
    voltages = np.ones(saturn.num_channels, dtype=c_float)
    saturn.reset_all_channels()
    saturn.get_channels()
    print("Setting all channel voltages to 10V")
    saturn.set_channels(voltages * 10/250)
    saturn.get_channels()
    print("Setting all channel voltages to 120V")
    saturn.set_channels(voltages * 120/250)
    saturn.get_channels()
    print("Setting all channel voltages to 245V")
    saturn.set_channels(voltages * 245/250)
    saturn.get_channels()
    print("\nTest complete.")
    # now let's test all the channels one by one.
    input("Press ENTER to test all channels individually\n")
    saturn.reset_all_channels()
    for i in range(saturn.num_channels):
        saturn.get_single_channel(i)
        saturn.set_single_channel(10/250, i)
        saturn.get_single_channel(i)
        saturn.set_single_channel(120/250, i)
        saturn.get_single_channel(i)
        saturn.set_single_channel(245/250, i)
        saturn.get_single_channel(i)
        saturn.reset_channel(i)
    print("\nTest complete.")
    input("\nPress ENTER to disconnect the mirror\n")
    saturn.disconnect_mirror()


