#include "mirror_wrapper.h"
#include "../cpp_src/Header/mirrorDriver.h"
#include "../cpp_src/Header/ws-util.h"


#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;
// This ensures that the C++ mirrorDriver class is used internally
// The MirrorHandle will be a pointer to a mirrorDriver instance

DLL_EXPORT MirrorHandle Mirror_Create() {
    mirrorDriver* md = new mirrorDriver();
    return static_cast<MirrorHandle>(md);
    //if (md == NULL)
    //{
    //    cout<<"*********************************************"<<endl;
    //}
    //return md;
}

DLL_EXPORT void Mirror_Destroy(MirrorHandle handle) {
    if (handle) {
        mirrorDriver* md = static_cast<mirrorDriver*>(handle);
        if (md->isConnected()) {
            md->disconnect();
        }
        delete md;
    }
}

DLL_EXPORT int Mirror_Connect(MirrorHandle handle) {
    if (!handle) return -1;
    return static_cast<mirrorDriver*>(handle)->connectToMirror();
}

DLL_EXPORT int Mirror_Disconnect(MirrorHandle handle) {
    if (!handle) return -1;
    return static_cast<mirrorDriver*>(handle)->disconnect();
}

DLL_EXPORT int Mirror_IsConnected(MirrorHandle handle) {
    if (!handle) return 0;
    return static_cast<mirrorDriver*>(handle)->isConnected();
}

DLL_EXPORT int Mirror_SetChannels(MirrorHandle handle, float* pattern_data) {
    if (!handle || !pattern_data) return -1;
    return static_cast<mirrorDriver*>(handle)->setChannels(pattern_data);
}

DLL_EXPORT int Mirror_SetSingleChannel(MirrorHandle handle, float value, int chid) {
    if (!handle) return -1;
    return static_cast<mirrorDriver*>(handle)->setSingleChannel(value, chid);
}

DLL_EXPORT const char* Mirror_GetServerIPAddress(MirrorHandle handle) {
    if (!handle) return NULL;
    return static_cast<mirrorDriver*>(handle)->getServerIPAddress();
}

DLL_EXPORT int Mirror_ResetAllChannels(MirrorHandle handle) {
    if (!handle) return -1;
    return static_cast<mirrorDriver*>(handle)->resetAllChans();
}

DLL_EXPORT int Mirror_ResetChannel(MirrorHandle handle, int chid) {
    if (!handle) return -1;
    return static_cast<mirrorDriver*>(handle)->resetChan(chid);
}

DLL_EXPORT int Mirror_GetChannelsStatus(MirrorHandle handle, float* arr) {
    if (!handle) return -1;
    return static_cast<mirrorDriver*>(handle)->getChannelsStatus(arr);
}

DLL_EXPORT int Mirror_GetNumChannels(MirrorHandle handle) {
    if (!handle) return -1;
    return static_cast<mirrorDriver*>(handle)->getNumMirrorChannels();
}

DLL_EXPORT float Mirror_GetSingleChannelStatus(MirrorHandle handle, int chid) {
    if (!handle) return -1;
    return static_cast<mirrorDriver*>(handle)->getSingleChannelStatus(chid);
}

DLL_EXPORT float Mirror_GetDriverVersion(MirrorHandle handle) {
    if (!handle) return -1;
    return static_cast<mirrorDriver*>(handle)->getDriverVersion();
}

DLL_EXPORT int Mirror_ConfigServerIPAddress(MirrorHandle handle, const char* ip_address) {
    if (!handle || !ip_address) return -1;
    // The underlying mirrorDriver::configIPAddress takes char*.
    // const_cast is used here assuming the function will not modify the input string
    // if it is only reading the IP. If it modifies it, the Python side should pass a mutable buffer.
    return static_cast<mirrorDriver*>(handle)->configIPAddress(const_cast<char*>(ip_address));
}

DLL_EXPORT int Mirror_ResetMirrorDriver(MirrorHandle handle) {
    if (!handle) return -1;
    return static_cast<mirrorDriver*>(handle)->resetMirrorDriver();
}
//DLL_EXPORT int Mirror_ApplyFullPattern(MirrorHandle handle, float* pattern_data, int num_elements) {
//    if (!handle || !pattern_data) return -1;
//    mirrorDriver* md = static_cast<mirrorDriver*>(handle);
//    return md->applyFullPattern(pattern_data);
//}

//DLL_EXPORT int Mirror_ApplySingleChannel(MirrorHandle handle, short channel, float value) {
//    if (!handle) return -1;
//    return static_cast<mirrorDriver*>(handle)->applySingleChannel(channel, value);
//}

// DLL_EXPORT int Mirror_GetMirrorPattern(MirrorHandle handle, float* buffer, int num_elements) {
//    if (!handle || !buffer) return -1;
//    mirrorDriver* md = static_cast<mirrorDriver*>(handle);
//    return md->getMirrorPattern(buffer);
// }



DLL_EXPORT const char* Mirror_GetLastErrorMessage( const char* prefix_message) {
    // WSAGetLastErrorMessage(const char* pcMessagePrefix, int nErrorID = 0)
    // Passing 0 for nErrorID usually means get the last error.
    return WSAGetLastErrorMessage(prefix_message, 0);
}