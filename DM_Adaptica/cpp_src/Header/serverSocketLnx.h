/***********************************************************************

  serverSocketLnx.h

  winsock functs porting on Linux for atomServer

  author: riccardo marogna for ADAPTICA 

  copyright (c) ADAPTICA 2008

***********************************************************************/


#ifndef SERVERSOCKETLNX_H_
#define SERVERSOCKETLNX_H_


#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <string>
#include <arpa/inet.h>


///////////////////////////////////////////////////////////////
// here we rewrite the winsock functs for Linux

//
#define INVALID_SOCKET 	-1
#define SOCKET_ERROR 	-1
#define SOCKET_CLOSED    0
#define SD_SEND 	SHUT_WR


// my socket (is simply an int...)
typedef int SOCKET;


int		closesocket(SOCKET m_sock);


#endif /*SERVERSOCKETLNX_H_*/
