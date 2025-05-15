""" This program was written by Matias Marambio in 2024.
It applies zonal wavefront correction with DMP-40 and WFS-20-5C
Any questions please guide them to mimarambio@uc.cl . EN-SPA-JP OK!
"""


import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ctypes
from ctypes import *


# Global Variables:

### DMP40 dlls ###
lib = cdll.LoadLibrary("C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLDFM_64.dll")
libX = cdll.LoadLibrary("C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLDFMX_64.dll")

### WFS dll ###
libwfs = cdll.LoadLibrary("C:\Program Files\IVI Foundation\VISA\Win64\Bin\WFS_64.dll")

### settling_time for interaction matrix ###
tilt_arm_time = 5e-1

### DM Voltages ###
segment_voltages = np.ones(40, dtype=float)*100
tilt_voltages = np.ones(3, dtype=float)*100

### WFS spot deviations ###
deviationsX = np.zeros((80, 80), dtype=np.float32)
deviationsY = np.zeros((80, 80), dtype=np.float32)

### Instrument Handles for SDK ###
dm_handle = c_ulong()
wfs_handle = c_ulong()

def initialize_wfs(valid_subaperture_value=64):
    """This function initializes the wavefront sensor. 
    The argument valid_subaperture_value refers to the flux intensity in a subaperture
    for it to be considered valid.
    Inside this function you can find the variables for the pupil. Please check that the values are correctly input
    depending on your use case.
    You can also set the black level offset and the exposure time.

    Returns:
     wfs_handle, num_spots_x, num_spots_y, pupil_x, pupil_y, pupil_diax, pupil_diay, intensityMask
    """
    global wfs_handle
    num_devices = c_int32()
    libwfs.WFS_GetInstrumentListLen(None, byref(num_devices))

    #if no devices connected, close the program
    if num_devices == 0:
        print("No availble devices.... closing program")
        quit()
    #get connection information for first available WFS device
    device_id = c_int32()
    device_in_use = c_int32() 
    device_name = create_string_buffer(20)
    serial_number = create_string_buffer(20)
    resource_name = create_string_buffer(30)
    libwfs.WFS_GetInstrumentListInfo(None, 0, byref(device_id), byref(device_in_use), device_name, serial_number, resource_name)
    #check if WFS is in use, if not, connect to device
    if device_in_use:
        print("Wavefront sensor currently in use.... closing program")
        quit()
    libwfs.WFS_init(resource_name, c_bool(False), c_bool(True), byref(wfs_handle))
    print(f"Connected to {device_name.value} with Serial Number {serial_number.value}")

    #Get the number of calibrated microlens arrays and print out data
    mla_count = c_int32()
    libwfs.WFS_GetMlaCount(wfs_handle, byref(mla_count))

    mla_name = create_string_buffer(20)
    cam_pitch = c_double()
    lenslet_pitch = c_double()
    spot_offset_x = c_double()
    spot_offset_y = c_double()
    lenslet_focal_length = c_double()
    astigmatism_correction_0 = c_double()
    astigmatism_correction_45 = c_double()
 
    print("Available Microlens Arrays: ")
    for i in range(mla_count.value):
        libwfs.WFS_GetMlaData(wfs_handle, i, mla_name, byref(cam_pitch), byref(lenslet_pitch), byref(spot_offset_x), 
            byref(spot_offset_y), byref(lenslet_focal_length), byref(astigmatism_correction_0), byref(astigmatism_correction_45))
        print(f"\tIndex: {i} - MLA Name: {mla_name.value} with lenslet pitch {lenslet_pitch.value}")

    #select MLA    
    libwfs.WFS_SelectMla(wfs_handle, 0)

    #configure cam resolution and pixel format. Method outputs number of spots in the X and Y for selected MLA
    # PIXEL_FORMAT_MONO8 = 0
    # CAM_RES_1280 = 0
    #Full lists of available sensor reolutions are in the WFS.h header file in C:\Program Files (x86)\IVI Foundation\VISA\WinNT\Include
    num_spots_x = c_int32()
    num_spots_y = c_int32()
    libwfs.WFS_ConfigureCam(wfs_handle, c_int32(0), c_int32(0), byref(num_spots_x), byref(num_spots_y))
    print(f"Number of detectable spots in X: {num_spots_x.value} \nNumber of detectable spots in Y: {num_spots_y.value}")

    #set WFS internal reference plane
    #other user-defined reference planes can be configured in the WFS software. These are saved to a .ref file and are accessed by passing a 1 instead of 0
    libwfs.WFS_SetReferencePlane(wfs_handle, c_int32(0))

    # TODO: make it so the user can define the pupil parameters in a txt file.
    pupil_x, pupil_y = c_double(.194), c_double(-.096)
    pupil_diax, pupil_diay = c_double(3.021), c_double(3.021)
    print("Define pupil to:")
    libwfs.WFS_SetPupil(wfs_handle, pupil_x, pupil_y, pupil_diax, pupil_diay)
    print(f"\tcenter = ({pupil_x.value:.3f}, {pupil_y.value:.3f})\n\tradius = {pupil_diax.value/2:.3f} mm.")

    #Take a series of images until one is usable. Check the device status after each image to determine usability
    # You need to start the WFS with this process or it will not work
    actual_exposure = c_double()
    actual_gain = c_double()
    device_status = c_int32()
    for i in range(10):
        libwfs.WFS_TakeSpotfieldImageAutoExpos(wfs_handle, byref(actual_exposure), byref(actual_gain))
        libwfs.WFS_GetStatus(wfs_handle, byref(device_status))
        if device_status.value & 0x00000002:
            print("Power too high")
        elif device_status.value & 0x00000004:
            print("Power too low")
        elif device_status.value & 0x00000008:
            print("High ambient light")
        else:
            print("Image is usable.... breaking loop")
            break
    
    #close program if image is not usable
    if device_status.value & 0x00000002 or device_status.value & 0x00000004 or device_status.value & 0x00000008:
        print("Image is not usable.... closing program")
        quit()

    #TODO: make it so the user can define the exposure time manually in a txt file
    #Take a series of images until one is usable. Check the device status after each image to determine usability

    libwfs.WFS_SetTriggerMode(wfs_handle, c_int32(0))
    libwfs.WFS_SetBlackLevelOffset(wfs_handle, c_int32(64))
    expt = c_double()
    libwfs.WFS_SetExposureTime(wfs_handle, c_double(84.), byref(expt))
    print("WFS exposure time:", expt.value) # This sometimes prints 0.0 ms, might be a problem with SDK. The real exposure time is whatever number is in the second argument.
    
    # Get mask based on intensity of each spot
    intensities = np.zeros((80, 80), dtype=np.float32)
    libwfs.WFS_CalcSpotsCentrDiaIntens(wfs_handle, c_int32(0), c_int32(0))
    libwfs.WFS_GetSpotIntensities(wfs_handle, intensities.ctypes.data_as(ctypes.POINTER(c_float)))

    return wfs_handle, num_spots_x, num_spots_y, pupil_x, pupil_y, pupil_diax, pupil_diay, intensities > valid_subaperture_value

def close_wfs():
    print('Closing WFS')
    libwfs.WFS_close(wfs_handle)

def initialize_dm(relax=False):
    """This function initializes the deformable mirror.
    You can choose to relax the deformable mirror in this function.

    Returns:
     dm_handle, segmentCount, tiltCount, minSegmentVoltage, maxSegmentVoltage
    """
    # Detect and initialize DMP40 device
    instrumentHandle = c_ulong()
    IDQuery = True
    resetDevice = False
    resource = c_char_p(b"")
    deviceCount = c_int()

    # Check how many DMP40 are connected
    lib.TLDFM_get_device_count(instrumentHandle, byref(deviceCount))
    if deviceCount.value < 1 :
        print("No DMP40 device found.")
        exit()
    else:
        print(deviceCount.value, "DMP40 device(s) found.\n")

    # Connect to the first available DMP40
    # Use TLDFMX_init to use functions in the extended driver as well
    lib.TLDFM_get_device_information(instrumentHandle, 0, 0, 0, 0, 0, resource)

    if (0 == libX.TLDFMX_init(resource.value, IDQuery, resetDevice, byref(instrumentHandle))):
        print("Connection to first DMP40 initialized.\n")
    else:
        print("Error with initialization.")
        exit()
    # Determine how many segments the mirror has and how many tilt arms.
    segmentCount = c_uint32()
    lib.TLDFM_get_segment_count(instrumentHandle, byref(segmentCount))
    print("Segment count:", segmentCount.value)
    tiltCount = c_uint32()
    lib.TLDFM_get_tilt_count(instrumentHandle, byref(tiltCount))
    print("Tilt count:", tiltCount.value)
    # Set voltages to a flat wavefront
    lib.TLDFM_set_segment_voltages(instrumentHandle, segment_voltages.ctypes.data_as(ctypes.POINTER(c_float)))
    lib.TLDFM_set_tilt_voltages(instrumentHandle, tilt_voltages.ctypes.data_as(ctypes.POINTER(c_float)))
    if relax:
        # Relax DMP40
        # devicePart determines which part of the mirror is relaxed
        # 0: only mirror, 1: only bimorph tilt arms, 2: both
        devicePart = c_uint32(2)
        isFirstStep = c_bool(True)
        reload = c_bool(False)
        # Create arrays for the mirror segment and tilt arm patterns
        relaxPatternMirror = (c_double*(segmentCount.value))()
        relaxPatternArms = (c_double*(tiltCount.value))()

        remainingSteps = c_int32()
        counter = 1
        # First relax step.
        print("Relaxing the DMP40.")
        print()
        libX.TLDFMX_relax(instrumentHandle, devicePart, isFirstStep, reload,
                     relaxPatternMirror, relaxPatternArms, byref(remainingSteps))

        lib.TLDFM_set_segment_voltages(instrumentHandle, relaxPatternMirror)
        lib.TLDFM_set_tilt_voltages(instrumentHandle, relaxPatternArms)
        print("\tRelax step:", counter)
        counter += 1

        isFirstStep = c_bool(False)

        # The following relax steps are made in a loop until the relaxation is complete.
        while remainingSteps.value > 0:
            libX.TLDFMX_relax(instrumentHandle, devicePart, isFirstStep, reload,
                     relaxPatternMirror, relaxPatternArms, byref(remainingSteps))
            lib.TLDFM_set_segment_voltages(instrumentHandle, relaxPatternMirror)
            lib.TLDFM_set_tilt_voltages(instrumentHandle, relaxPatternArms)
            print("\tRelax step:", counter)
            counter += 1
        print("Relaxing complete.")
    # Obtain maximum and minimum segment voltage and tilt voltage for poke matrix creation
    minSegmentVoltage, maxSegmentVoltage, segmentCommonVoltageMax, minTiltVoltage, maxTiltVoltage, tiltCommonVoltageMax = (
           c_double(),        c_double(),              c_double(),     c_double(),     c_double(),           c_double())
    lib.TLDFM_get_device_configuration(instrumentHandle, byref(segmentCount), byref(minSegmentVoltage),
                                       byref(maxSegmentVoltage), byref(segmentCommonVoltageMax), byref(tiltCount),
                                       byref(minTiltVoltage), byref(maxTiltVoltage), byref(tiltCommonVoltageMax))
    print(f"""Voltage range information:
    Min. segment voltage = {minSegmentVoltage.value}V
    Max. segment voltage = {maxSegmentVoltage.value}V
    Min. tip-tilt voltage = {minTiltVoltage.value}V
    Max. tip-tilt voltage = {maxTiltVoltage.value}V""")
    return instrumentHandle, segmentCount, tiltCount, minSegmentVoltage, maxSegmentVoltage

def close_dm():
    print("Closing DM.")
    lib.TLDFM_close(dm_handle)

def enable_hysteresis_correction(instrumentHandle):
    """This function is used to enable hysteresis correction in the deformable mirror.
    It has not been properly tested yet.
    """
    # Enable hysteresis compensation for DMP40
    mirror_hystcomp, tt_hystcomp = c_bool(), c_bool()
    print("checking hysteresis compensation...")
    lib.TLDFM_enabled_hysteresis_compensation(instrumentHandle, c_uint32(0), byref(mirror_hystcomp))
    lib.TLDFM_enabled_hysteresis_compensation(instrumentHandle, c_uint32(1), byref(tt_hystcomp))
    if not mirror_hystcomp.value and not tt_hystcomp.value:
        print("\thysteresis compensation not enabled.")
        print("\tenabling hysteresis compensation for both mirror and tip-tilt.")
        lib.TLDFM_enable_hysteresis_compensation(instrumentHandle, c_uint32(2), c_bool(1))
    elif not mirror_hystcomp.value:
        print("\thysteresis compensation already enabled for tip-tilt.")
        print("\tenabling hysteresis compensation for mirror.")
        lib.TLDFM_enable_hysteresis_compensation(instrumentHandle, c_uint32(0), c_bool(1))
    elif not mtt_hystcomp.value:
        print("\thysteresis compensation already enabled for mirror.")
        print("\tenabling hysteresis compensation for tip-tilt.")
        lib.TLDFM_enable_hysteresis_compensation(instrumentHandle, c_uint32(1), c_bool(1))
    return

def get_wfs_image():
    """This function outputs a video feed of the wavefront sensor spot field.
    """
    print("Showing spot field image...")
    frame = np.zeros((1440, 1080), dtype=np.uint8)
    imgw = c_int32()
    imgh = c_int32()

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    libwfs.WFS_TakeSpotfieldImage(wfs_handle)
    libwfs.WFS_GetSpotfieldImageCopy(wfs_handle, frame.ctypes.data_as(ctypes.POINTER(c_int32)), byref(imgh), byref(imgw))
    img = ax.imshow(frame.T, cmap='gray', animated=True)
    plt.title('WFS Video Feed')

    # Function to update the plot for each frame
    def update(frame_number):
        libwfs.WFS_TakeSpotfieldImage(wfs_handle)
        libwfs.WFS_GetSpotfieldImageCopy(wfs_handle, frame.ctypes.data_as(ctypes.POINTER(c_int32)), byref(imgh), byref(imgw))
        img.set_array(frame)
        return img,
    expt = c_double()
    libwfs.WFS_GetExposureTime(wfs_handle, byref(expt))
    print(f"The exposure time in milliseconds is: {expt.value:.2f}")
    # Set up the animation
    animation = FuncAnimation(fig, update, frames=100, interval=expt.value, blit=True)
    print(f"Close the plot to continue.")
    plt.show()

def tilt_mirror(arm, voltage):
    print(f"Tilting arm {arm} to {voltage:.2f}V")
    lib.TLDFM_set_tilt_voltage(dm_handle, c_uint32(arm), c_double(voltage))
    return

def move_segment(segment, voltage):
    print(f"Moving segment {segment} to {voltage:.2f}V")
    lib.TLDFM_set_segment_voltage(dm_handle, c_uint32(segment), c_double(voltage))
    return

def build_interaction_matrix(num_spots_x, num_spots_y, segmentCount, tiltCount, mask, reps=3):
    r"""Note: Interaction matrix has the same meaning as Poke matrix
    This function builds the interaction matrix between the WFS and the DM via the poking method.
    You need to specify the number of spots, the number of tilt arms and segments, the intensity
    mask and the number of repetitions of push/pull for each segment/arm.
    This functions saves an image of the interaction matrix here: C:\Users\AO\source\repos\ao-python\interaction_matrix.png

    Returns:
    interaction matrix
    """
    # Get the deviations in a numpy array
    deviationsX = np.zeros((80, 80), dtype=np.float32)
    deviationsY = np.zeros((80, 80), dtype=np.float32)
    n_valid_subap = np.sum(mask)
    print(f"Number of valid subapertures: {n_valid_subap}")
    aux_x = np.zeros((n_valid_subap, reps), dtype=np.float32)
    aux_y = np.zeros((n_valid_subap, reps), dtype=np.float32)
    x_interaction_matrix = np.zeros((n_valid_subap,
                                segmentCount.value + tiltCount.value), dtype=np.float32)
    y_interaction_matrix = np.zeros((n_valid_subap,
                                segmentCount.value + tiltCount.value), dtype=np.float32)
    try:
        for axis in range(tiltCount.value):
            for rep in range(reps):
                # push
                tilt_mirror(axis, tilt_voltages[axis]+5.)
                time.sleep(tilt_arm_time)
                libwfs.WFS_TakeSpotfieldImage(wfs_handle)
                libwfs.WFS_CalcSpotsCentrDiaIntens(wfs_handle, c_int32(1), c_int32(0))
                libwfs.WFS_CalcSpotToReferenceDeviations(wfs_handle, c_int32(0))
                libwfs.WFS_GetSpotDeviations(wfs_handle,
                                        deviationsX.ctypes.data_as(ctypes.POINTER(c_float)),
                                        deviationsY.ctypes.data_as(ctypes.POINTER(c_float))
                                        )
                aux_x[:, rep] = deviationsX[mask]
                aux_y[:, rep] = deviationsY[mask]
                # pull
                tilt_mirror(axis, tilt_voltages[axis]-5.)
                time.sleep(tilt_arm_time)
                libwfs.WFS_TakeSpotfieldImage(wfs_handle)
                libwfs.WFS_CalcSpotsCentrDiaIntens(wfs_handle, c_int32(1), c_int32(0))
                libwfs.WFS_CalcSpotToReferenceDeviations(wfs_handle, c_int32(0))
                libwfs.WFS_GetSpotDeviations(wfs_handle,
                                        deviationsX.ctypes.data_as(ctypes.POINTER(c_float)),
                                        deviationsY.ctypes.data_as(ctypes.POINTER(c_float))
                                        )
                aux_x[:, rep] -=  deviationsX[mask]
                aux_y[:, rep] -=  deviationsY[mask]
                x_interaction_matrix[:, axis] = np.nanmean(aux_x/10., axis=1)
                y_interaction_matrix[:, axis] = np.nanmean(aux_y/10., axis=1)
            # reset
            tilt_mirror(axis, tilt_voltages[axis])
        for s in range(segmentCount.value):
            for rep in range(reps):
                # push
                move_segment(s, segment_voltages[s]+5.)
                time.sleep(tilt_arm_time)
                libwfs.WFS_TakeSpotfieldImage(wfs_handle)
                libwfs.WFS_CalcSpotsCentrDiaIntens(wfs_handle, c_int32(1), c_int32(0))
                libwfs.WFS_CalcSpotToReferenceDeviations(wfs_handle, c_int32(0))
                libwfs.WFS_GetSpotDeviations(wfs_handle,
                                        deviationsX.ctypes.data_as(ctypes.POINTER(c_float)),
                                        deviationsY.ctypes.data_as(ctypes.POINTER(c_float))
                                        )
                aux_x[:, rep] = deviationsX[mask]
                aux_y[:, rep] = deviationsY[mask]
                # pull
                move_segment(s, segment_voltages[s]-5.)
                time.sleep(tilt_arm_time)
                libwfs.WFS_TakeSpotfieldImage(wfs_handle)
                libwfs.WFS_CalcSpotsCentrDiaIntens(wfs_handle, c_int32(1), c_int32(0))
                libwfs.WFS_CalcSpotToReferenceDeviations(wfs_handle, c_int32(0))
                libwfs.WFS_GetSpotDeviations(wfs_handle,
                                        deviationsX.ctypes.data_as(ctypes.POINTER(c_float)),
                                        deviationsY.ctypes.data_as(ctypes.POINTER(c_float))
                                        )
                aux_x[:, rep] -=  deviationsX[mask]
                aux_y[:, rep] -=  deviationsY[mask]
                x_interaction_matrix[:, s+3] = np.nanmean(aux_x/10., axis=1)
                y_interaction_matrix[:, s+3] = np.nanmean(aux_y/10., axis=1)
            # reset
            move_segment(s, segment_voltages[s])
    except:
        close_dm()
        close_wfs()
        print("Interaction matrix tilt nan error")
        sys.exit()
    plt.matshow(np.vstack((x_interaction_matrix, y_interaction_matrix)), aspect="auto")
    plt.colorbar()
    plt.savefig(r"C:\Users\AO\source\repos\ao-python\interaction_matrix.png")
    plt.clf()
    #plt.show()
    imat = np.vstack((x_interaction_matrix, y_interaction_matrix))
    return imat

def get_reconstruction_matrix(B, n_modes):
    """This function computes the reconstruction matrix given an interaction matrix and the number
    of modes wanted (from lowest to highest), via the truncated SVD method.

    Returns:
    Reconstruction matrix
    """

    S = np.zeros(B.shape, dtype=float)
    U, S_aux, Vt = np.linalg.svd(B)
    V = Vt.T
    np.fill_diagonal(S, S_aux)
    S_inv = np.linalg.pinv(S)
    G = np.zeros(V.shape, dtype=float)
    G_aux = np.zeros(V.shape[0], dtype=float)
    G_aux[0:np.min((n_modes, V.shape[0]))] = 1
    np.fill_diagonal(G, G_aux)
    R = V @ G @ S_inv @ U.T
    
    return R

def correct_wavefront(s, R, gain, only_tilts=False, segment=False):
    """This function calculates the voltage instructions and applies them to the arms and segments of the deformable mirror.
    The function truncates the voltage of the deformable mirror at 195V and 5V.
    """

    global segment_voltages, tilt_voltages
    a = R @ s

    lib.TLDFM_get_measured_segment_voltages(dm_handle, segment_voltages.ctypes.data_as(ctypes.POINTER(c_float)))
    lib.TLDFM_get_measured_tilt_voltages(dm_handle, tilt_voltages.ctypes.data_as(ctypes.POINTER(c_float)))

    underMask = tilt_voltages - a[:3] * gain < 5
    overMask = tilt_voltages - a[:3] * gain > 195
    tilt_voltages = tilt_voltages - a[:3].astype(float) * gain

    if np.any(underMask):
        print("Warning, tilt arm instruction voltage is lower than 5V. Truncating to 5V")
        tilt_voltages[underMask] = 5.
    if np.any(overMask):
        print("Warning, tilt arm instruction voltage is higher than 195V. Truncating to 195V")
        tilt_voltages[overMask] = 195.

    lib.TLDFM_set_tilt_voltages(dm_handle, tilt_voltages.ctypes.data_as(ctypes.POINTER(c_float)))
    print("Tilt Voltages:")
    print(tilt_voltages)

    underMask = segment_voltages - a[3:] * gain < 5
    overMask = segment_voltages - a[3:] * gain > 195
    segment_voltages = segment_voltages - a[3:].astype(float) * gain

    if np.any(underMask):
        print("Warning, actuator instruction voltage is lower than 5V. Truncating to 5V")
        segment_voltages[underMask] = 5.
    if np.any(overMask):
        print("Warning, actuator instruction voltage is higher than 195V. Truncating to 195V")
        segment_voltages[overMask] = 195.
    lib.TLDFM_set_segment_voltages(dm_handle, segment_voltages.ctypes.data_as(ctypes.POINTER(c_float)))
    print("Segment Voltages:")
    print(segment_voltages)

def plot_slopes(slopesX, slopesY, num_spots_x, num_spots_y, mask=None):
    """This function plots the slopes as vector plot with a normalized magnitude background.
    """
    if mask is None:
        mask = np.ones(slopesX.shape)
    slopesX[~mask], slopesY[~mask] = np.nan, np.nan
    X, Y = np.meshgrid(range(num_spots_x), range(num_spots_y))
    plt.imshow(np.hypot(slopesX[:num_spots_y, :num_spots_x], slopesY[:num_spots_y, :num_spots_x]), cmap="viridis", interpolation="nearest")
    plt.quiver(X, Y, slopesX[:num_spots_y, :num_spots_x], slopesY[:num_spots_y, :num_spots_x], color="k", headwidth=2)
    plt.colorbar()
    plt.title(f"Slopes X {np.nanmean(slopesX):.2f} | Slopes Y {np.nanmean(slopesY):.2f}")
    plt.show()

def reset_dev_matrix(valid_subaperture_value=64):
    """This function "cleans" the deviations matrix, considering that some subapertures could have not enough
    flux to be considered in the next calculation. It returns a mask that needs to be applied just after taking a new
    wfs spotfield image.

    Returns:
     mask
    """
    deviationsX = np.zeros((80, 80), dtype=np.float32)
    deviationsY = np.zeros((80, 80), dtype=np.float32)
    # Get mask based on intensity of each spot
    intensities = np.zeros((80, 80), dtype=np.float32)
    libwfs.WFS_CalcSpotsCentrDiaIntens(wfs_handle, c_int32(1), c_int32(0))
    libwfs.WFS_GetSpotIntensities(wfs_handle, intensities.ctypes.data_as(ctypes.POINTER(c_float)))
    mask = intensities > valid_subaperture_value
    return mask

def main():
    """Main function
    """
    global tilt_voltages, segment_voltages, deviationsX, deviationsY, dm_handle, wfs_handle

    wfs_handle, num_spots_x, num_spots_y, pupil_x, pupil_y, pupil_diax, pupil_diay, mask = initialize_wfs()
    dm_handle, segmentCount, tiltCount, minVoltage, maxVoltage = initialize_dm()
    tmode = c_int32()
    libwfs.WFS_GetTriggerMode(wfs_handle, byref(tmode))
    print(f"Trigger mode: {tmode.value}")
    B = build_interaction_matrix(num_spots_x, num_spots_y, segmentCount, tiltCount, mask)
    print("Interaction Matrix:")
    print(B)
    try:
        n = int(input("Choose the number of modes you want to correct (default, 20).\n\t"))
    except:
        n = 20
    try:
        reps = int(input("Choose the number of repetitions of the loop (default, 1000).\n\t"))
    except:
        reps = 1000
    # Now that we have the interaction matrix, we can build the reconstruction matrix
    R = get_reconstruction_matrix(B, n)
    for i in range(reps):
        # Get the deviations in a numpy array
        deviationsX = np.zeros((80, 80), dtype=np.float32)
        deviationsY = np.zeros((80, 80), dtype=np.float32)
        libwfs.WFS_TakeSpotfieldImage(wfs_handle)
        nanmask = reset_dev_matrix()
        libwfs.WFS_CalcSpotToReferenceDeviations(wfs_handle, c_int32(0))
        libwfs.WFS_GetSpotDeviations(wfs_handle,
                            deviationsX.ctypes.data_as(ctypes.POINTER(c_float)),
                            deviationsY.ctypes.data_as(ctypes.POINTER(c_float))
                            )
        deviationsX[~nanmask], deviationsY[~nanmask] = 0, 0
        slopes = np.hstack((deviationsX[mask], deviationsY[mask]))
        # sometimes even if a subaperture has enough flux to pass the reset_dev_matrix function,
        # it will still end up as nan, so:
        if np.any(np.isnan(slopes)):
            slopes = np.nan_to_num(slopes)
        correct_wavefront(slopes, R, 0.45)
        print(f"----iter {i+1:04d}----")
    # Get the deviations in a numpy array
    libwfs.WFS_TakeSpotfieldImage(wfs_handle)
    nanmask = reset_dev_matrix()
    libwfs.WFS_CalcSpotsCentrDiaIntens(wfs_handle, c_int32(1), c_int32(0))
    libwfs.WFS_CalcSpotToReferenceDeviations(wfs_handle, c_int32(0))
    libwfs.WFS_GetSpotDeviations(wfs_handle,
                            deviationsX.ctypes.data_as(ctypes.POINTER(c_float)),
                            deviationsY.ctypes.data_as(ctypes.POINTER(c_float))
                            )
    deviationsX[~nanmask], deviationsY[~nanmask] = 0, 0
    slopes = np.hstack((np.nanmean(deviationsX[mask]), np.nanmean(deviationsY[mask])))
    print(slopes)
    close_wfs()
    close_dm()
    plot_slopes(deviationsX, deviationsY, num_spots_x.value, num_spots_y.value, mask)

if __name__ == "__main__":
    main()
