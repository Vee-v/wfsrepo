#ifndef MIRROR_WRAPPER_H
#define MIRROR_WRAPPER_H

// Define DLL_EXPORT

#define DLL_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef void* MirrorHandle; // Opaque handle to the mirrorDriver object

DLL_EXPORT MirrorHandle Mirror_Create();
DLL_EXPORT void Mirror_Destroy(MirrorHandle handle);

DLL_EXPORT int Mirror_Connect(MirrorHandle handle);
DLL_EXPORT int Mirror_Disconnect(MirrorHandle handle);
DLL_EXPORT int Mirror_IsConnected(MirrorHandle handle);

DLL_EXPORT int Mirror_SetChannels(MirrorHandle handle, float* pattern_data);
DLL_EXPORT int Mirror_SetSingleChannel(MirrorHandle handle, float value, int chid);
DLL_EXPORT const char* Mirror_GetServerIPAddress(MirrorHandle handle);
DLL_EXPORT int Mirror_ResetAllChannels(MirrorHandle handle);
DLL_EXPORT int Mirror_ResetChannel(MirrorHandle handle, int chid);
DLL_EXPORT int Mirror_GetChannelsStatus(MirrorHandle handle, float* arr);
DLL_EXPORT int Mirror_GetNumChannels(MirrorHandle handle);
DLL_EXPORT float Mirror_GetSingleChannelStatus(MirrorHandle handle, int chid);
DLL_EXPORT float Mirror_GetDriverVersion(MirrorHandle handle);
DLL_EXPORT int Mirror_ConfigServerIPAddress(MirrorHandle handle, const char* ip_address);
DLL_EXPORT int Mirror_ResetMirrorDriver(MirrorHandle handle);
DLL_EXPORT const char* Mirror_GetLastErrorMessage(const char* prefix_message);


#ifdef __cplusplus
}
#endif


#endif // MIRROR_WRAPPER_H

