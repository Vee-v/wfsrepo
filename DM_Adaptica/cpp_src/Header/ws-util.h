/***********************************************************************
	
	ws-util: various socket-com utility functions & declarations.
 
	author: riccardo marogna for ADAPTICA 

	copyright (c) ADAPTICA 2008

***********************************************************************/



#ifndef _WS_UTIL_H
#define _WS_UTIL_H

// debug flag
#ifndef DEBUG
#define DEBUG
#endif

// verbose debug flag
#ifndef VERBOSE
//#define VERBOSE
#endif


////////////////////////////////////////////////////
// windows uses winsock
#ifdef _WIN32
#include <winsock.h>
#include <windows.h>
#include <dos.h>
#include <direct.h>
//#include <winsock2.h>
#if !defined(_WINSOCK2API_) 
// Winsock 2 header defines this, but Winsock 1.1 header doesn't.  In
// the interest of not requiring the Winsock 2 SDK which we don't really
// need, we'll just define this one constant ourselves.
#define SD_SEND 1
#endif
/////////////////////////////////////////////////////

////////////////////////////////////////////////////
#else // header for linuz
#include "../Header/serverSocketLnx.h"
#include <pthread.h>
#include <signal.h>
#endif

///////////////////////////////////////////////////
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "Timer.h"


/*******************************************************/
// Default port to connect to on the server
#define MAIN_PORT 4242

// service port for the service thread
#define SERVICE_PORT 4243

/***********************************************************************/
// com_types between server/client 
#define GET_DRV_VERS    	100
#define GET_CHNUM		101
#define FULLCHTX		102
#define SINGLECHTX		103
#define FULLCHRX		104
#define SINGLECHRX		105
#define TESTNCHTX		106

/***********************************************************************/
// com_types for CLIENT-SUPERVISOR COM
#define SHUTDOWN                255
#define SRVR_REBOOT             254
#define IPCONF                  253
#define READY                   252
#define RESET_COM               251
#define SRVR_UPDATE             250
#define HANDSHAKE               249
#define READY_TO_EAT            248
#define SRVR_BIN_SEND           247

/***********************************************************************/
// reserved client id
#define CLIENT_ID                55

// reserved supervisor id
#define SRVRSUPER_ID             66

// reserved mirachServer id
#define MIRACH_SERVER_ID	 77


/***********************************************************************/
// data sizes
#define DRV_VERS_SIZE	4               // float
#define	VAL_SIZE		1				// byte
#define ID_SIZE			1	
#define CHNUM_SIZE		4				// int
#define SINGLECHTX_SIZE	3               // 1 byte for val, 2 bytes for num channel
#define NDATABYTES_SIZE 4               // size of number of data bytes (int)
#define IPADDR_LEN 15                   // IP address length
#define CODEW_SIZE 1                    // code word size (bytes)
#define CHECKSUM_SIZE 2                 // checksum size (bytes)
#define SERVICE_BUF_LEN 16              // service buffer (useless?)

/*************************************************************/
// DEFAULT VALS
#define CH_RESET_VAL    0   // default reset value for mirror channels
#define DEF_CH_NUM      64  // def num of channels
#define DRIVER_MAX_VAL  255 // max driving val for mirror channels set
#define INIT_SOCK       -99 
#define MAX_NUM_VERSS   100
#define CONNECTIONS_TIMEOUT 6 // timeout for waiting for socket connection (s) 





//////////////////////////////////////////////////////////////////////////////////
// in/out data struct
struct dataInOut
{
	// num of data bytes
	int	ndatabytes;
	
	// checkSum field for data check
	short	checksum;
	
	// command type
	unsigned char com_type;

	// pointer to data
	unsigned char* data;
};


/***** ETHERNET COM ROUTINES **********************************/


// accept connection from client
extern SOCKET    AcceptConnection(SOCKET ListeningSocket, sockaddr_in& sinRemote);
                

// sets up the listener, returning the socket
extern SOCKET    SetUpListener(const char* pcAddress, int nPort);
                

/***********************************************************************
// LookupAddress
// Given an address string, determine if it's a dotted-quad IP address
// or a domain address.  If the latter, ask DNS to resolve it.  In
// either case, return resolved IP address.  If we fail, we return
// INADDR_NONE.
***********************************************************************/
extern u_long    LookupAddress(const char* pcHost);


// shutdown the connection for a given socket
extern bool      ShutdownConnection(SOCKET sd);

/***********************************************************************
// EstablishConnection
// Connects to a given address, on a given port, both of which must be
// in network byte order.  Returns newly-connected socket if we succeed,
// or INVALID_SOCKET if we fail.
***********************************************************************/
extern SOCKET   EstablishConnection(u_long nRemoteAddr, u_short nPort);


// send buffer out
extern	int	sendBufferOut(dataInOut * md, SOCKET sd);


// recBufferIn : receive nBytes bytes in bufferIn
extern int      recBufferIn(int nBytes, unsigned char* buf_in, SOCKET sd);
        

// check the incoming buffer & write datafields
// args: pointer to data struct, pointer to bufferIn, socket.
extern int      dataCheckIn(dataInOut *d, SOCKET sd);


// checksum routines 
extern short	 calcSum(dataInOut *d);


extern short	 checkSum(dataInOut *d);


////////////////////////////////////////////
// thread for connections timeouts
void      *timerThread( void *condition );


////////////////////////////////////////////
// thread for connections timeouts
void   createDir( char*  dir );



/////////////////////////////////////////////
#ifdef WIN32
	// rewrite a linux funct for winz
	extern void sleep(int secs);
#endif

// winsock errors manager
extern const char* WSAGetLastErrorMessage(const char* pcMessagePrefix, int nErrorID=0);


#endif // !defined (WS_UTIL_H)

