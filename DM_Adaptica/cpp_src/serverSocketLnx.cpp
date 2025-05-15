/***********************************************************************

  serverSocketLnx.cpp

  author: riccardo marogna for ADAPTICA 

  copyright (c) ADAPTICA 2008

***********************************************************************/

#include "../Header/global.h"
#include "../Header/serverSocketLnx.h"


 ///////////////////////////////////////////////////////
// 
int closesocket(SOCKET m_sock)
{
	
	if (m_sock!=INVALID_SOCKET){ 
		::close ( m_sock );
		return SOCKET_CLOSED;
	}
	else 
		return SOCKET_ERROR; 
}


