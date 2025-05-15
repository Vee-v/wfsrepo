/***********************************************************************
	
	global: various declaration and definitions for compilations and etc....
 
	author: ADAPTICA 

	copyright (c) ADAPTICA 2008

***********************************************************************/

#ifndef _GLOBAL_H
#define _GLOBAL_H

// debug flag
#ifndef DEBUG
#define DEBUG
#endif

// verbose debug flag
#ifndef VERBOSE
//#define VERBOSE
#endif

// Activate this flag to use the MirachServer & MirachServerSupervisor outside or inside the Mirach
#ifndef NO_MIRACH
//#define NO_MIRACH
#endif

// Activate this flag to avoid server launching
#ifndef NO_LAUNCH_SERVER
//#define NO_LAUNCH_SERVER
#endif

// Activete this flag if you want to scale the output
#ifndef SCALING_ACTIVATE
    #define SCALING_ACTIVATE
    #define SCALING_VALUE       (0.9f)
#endif

// **********************
// * DEFAULT SERVER IP  *
// **********************
#ifndef NO_MIRACH
    #define DEF_SERVER_IP "192.168.0.1"
#else
    #define DEF_SERVER_IP "127.0.0.1"
#endif



// ******************
// * Server version *
// ******************
#define MIRACH_SERVER_VERS       4

// ******************************
// * Server Supervisor  version *
// ******************************
#define MIRACH_SUPER_VERS        4

// **************************************************
// * Cerver/Client config files (IP address etc)    *
// **************************************************
#define SERVER_CURRENT_VERS  "AOfiles/serverCurrentVers.aconf"
#define SERVER_IPCONFIG_FILE "AOfiles/serverIpConfig.aconf"
#define SERVER_DIR "AOfiles/"
#define DEF_SERVER_PROGRAM   "./mirachserver_v_01"



#endif // !defined (GLOBAL_H)

