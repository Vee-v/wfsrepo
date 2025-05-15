/***********************************************************************
 ws-util.cpp - Some basic Winsock utility functions.

***********************************************************************/

#include <iostream>
#include <algorithm>
#include <strstream>

#include "../Header/global.h"
#include "../Header/ws-util.h"
#ifndef _WIN32     
        #include <sys/stat.h>
#endif


using namespace std;


/*************************************************************************/
// SetUpListener
// Sets up a listener on the given interface and port, returning the
// listening socket if successful; if not, returns INVALID_SOCKET.
/*************************************************************************/
SOCKET SetUpListener(const char* pcAddress, int nPort)
{
   
    u_long nInterfaceAddr = inet_addr(pcAddress);
    //u_long nInterfaceAddr = INADDR_ANY;
    
	if (nInterfaceAddr != INADDR_NONE)
        {
            SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);

            if (sock != INVALID_SOCKET)
            {
                sockaddr_in sinInterface;
                sinInterface.sin_family = AF_INET;
                sinInterface.sin_addr.s_addr = nInterfaceAddr;
                sinInterface.sin_port = nPort;
                int tmp=bind(sock, (sockaddr*)&sinInterface,sizeof(sockaddr_in));

                cout << "SetupListener:bind  "<<tmp << endl;
		if (tmp!= SOCKET_ERROR)
                {
                    listen(sock, 1);
                    return sock;
                }

                printf("******* setUpListener: SOCKET ERROR!\n");
                    perror("binderror: "); 
					//in linux c'era solo close
					//close(sock);
					//siccome in win mi da errore lo rifaccio cos� con closesocket
                closesocket(sock);
            }
        }

       
 
        return INVALID_SOCKET;
}

/***********************************************************************
// LookupAddress
// Given an address string, determine if it's a dotted-quad IP address
// or a domain address.  If the latter, ask DNS to resolve it.  In
// either case, return resolved IP address.  If we fail, we return
// INADDR_NONE.
***********************************************************************/
u_long LookupAddress(const char* pcHost)
{
    u_long nRemoteAddr = inet_addr(pcHost);
    
    if (nRemoteAddr == INADDR_NONE) 
    {
        // pcHost isn't a dotted IP, so resolve it through DNS
        hostent* pHE = gethostbyname(pcHost);
        if (pHE == 0) 
        {
            return INADDR_NONE;
        }
        
        nRemoteAddr = *((u_long*)pHE->h_addr_list[0]);
    }

    return nRemoteAddr;
}


/***********************************************************************
// EstablishConnection
// Connects to a given address, on a given port, both of which must be
// in network byte order.  Returns newly-connected socket if we succeed,
// or INVALID_SOCKET if we fail.
***********************************************************************/
SOCKET EstablishConnection(u_long nRemoteAddr, u_short nPort)
{
    // Create a stream socket
    #ifdef DEBUG
    cout<<"EstablishConnection: creating socket...\n";
    #endif
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    
    if (sock != INVALID_SOCKET) 
    {
        #ifdef DEBUG
        cout<<"...done.\n";
        #endif
        sockaddr_in sinRemote;
        sinRemote.sin_family = AF_INET;
        sinRemote.sin_addr.s_addr = nRemoteAddr;
        sinRemote.sin_port = nPort;
        

        ///////////////////////////////////////////////////////////////////////////////
        // start a timer thread
        int connected=0;

#ifndef _WIN32
        pthread_t timer;
        pthread_create(&timer, NULL, timerThread, (void *)(&connected) );
#endif
        //////////////////////////////////////////////////////////////////////////////

        #ifdef DEBUG
        cout<<"connecting...\n";
        #endif
        if ( connect(sock,(sockaddr*)&sinRemote,sizeof(sockaddr_in))==SOCKET_ERROR ) 
        {
            // tell the timer to stop
            connected=-1;
            sock = INVALID_SOCKET;
        }
        else
        {
            // tell the timer to stop
            connected=1;
        }
#ifndef _WIN32     
        pthread_join(timer, NULL);
#endif
    }
    
    return sock;
}


/*************************************************************************/
// AcceptConnection
// Waits for a connection on the given socket.  When one comes in, we
// return a socket for it.  If an error occurs, we return
// INVALID_SOCKET.
/*************************************************************************/
extern SOCKET AcceptConnection(SOCKET ListeningSocket, sockaddr_in& sinRemote)
{
    int nAddrSize = sizeof(sinRemote);

#ifdef _WIN32
    return accept(ListeningSocket, (sockaddr*)&sinRemote, &nAddrSize);
#else
    return accept(ListeningSocket, (sockaddr*)&sinRemote, (socklen_t*)&nAddrSize);
#endif

}

/*************************************************************************/
// send function: compose bufferOut & send to host
// init bufferTxRx: 
// 1 bytes for com_type
// + 4 for num of following data bytes
// + data bytes
// + 2 for checksum 
/*************************************************************************/
extern int sendBufferOut(dataInOut *md, SOCKET sd)
{	
	// calc checksum
	md->checksum = calcSum(md);

	int bufLen = CODEW_SIZE + NDATABYTES_SIZE + md->ndatabytes + CHECKSUM_SIZE;
	char *bufferOut = new char[bufLen];
	
	*(bufferOut) = md->com_type;
	*(int*)(bufferOut + CODEW_SIZE) = md->ndatabytes;
	for (int i=0; i<md->ndatabytes; i++)
		*(bufferOut + CODEW_SIZE + NDATABYTES_SIZE + i) = (char)*(md->data+i);
	
	// write checksum 
	*(short*)(bufferOut + CODEW_SIZE + NDATABYTES_SIZE + md->ndatabytes) = calcSum(md);

	// num of bytes to send
	int nbytes = md->ndatabytes + CODEW_SIZE + NDATABYTES_SIZE + CHECKSUM_SIZE;

        ///////////////////////////////////////////////////////////////////////////////
        // start a timer thread to abort in case send function is not responding..
        int sent=0;
		#ifndef _WIN32
        pthread_t timer;
        pthread_create(&timer, NULL, timerThread, (void *)(&sent) );
		#endif
        //////////////////////////////////////////////////////////////////////////////
        
        
        /////////////////////////////////////////////////////////////////////
	// send on sock
        int nTemp=0;
        int nSentBytes=0;
        
        while (nTemp < nbytes)
        {
            // here send on socket
            nTemp = send(sd, (char*)(bufferOut + nSentBytes), nbytes - nSentBytes, 0);
            
            if (nTemp>0)
            {
                //cout<<nbytes<<" bytes<<"sent\n";
                nSentBytes += nTemp;
            }
            
            else if(nTemp == SOCKET_ERROR)
            {
                // signals to the timer thread
                sent = -1;
				#ifndef _WIN32
                pthread_join(timer, NULL);
				#endif
                delete[] bufferOut;
                return 0;
            }
            
            else
            {
                cout << "Peer unexpectedly dropped connection!" << endl;
                // signals to the timer thread
                sent = -1;
				#ifndef _WIN32
                pthread_join(timer, NULL);
				#endif
                delete[] bufferOut;
                return 0;
            }
	}
        
        // signals to the timer thread
        sent = 1;
		#ifndef _WIN32
        pthread_join(timer, NULL);
		#endif
        
	////////////////////////////////////////////////////////////////////
        delete[] bufferOut;
	return 1;
}


/*************************************************************************/
// recBufferIn : receive nBytes bytes in buf_in
/*************************************************************************/
extern int recBufferIn(int nBytes, unsigned char* buf_in, SOCKET sd)
{
    
	int nNewBytes,nRecBytes;
        
        // init data counters
	nRecBytes=nNewBytes=0;
        
         ///////////////////////////////////////////////////////////////////////////////
        //CANNOT USE A TIMER HERE !!
        // start a timer thread to abort in case connection is not responding..
        //int received=0;
        //pthread_t timer;
        //int t_ret = pthread_create(&timer, NULL, timerThread, (void *)(&received) );
        //////////////////////////////////////////////////////////////////////////////
        
	while (nRecBytes<nBytes)
        {
            nNewBytes = recv(sd, (char*)(buf_in + nRecBytes), nBytes - nRecBytes, 0);
            
            if (nNewBytes>0)
            {
                //cout<<"receiving "<<nNewBytes<<"bytes"<<endl;
                nRecBytes += nNewBytes;
            }
            
            else if(nNewBytes == SOCKET_ERROR)
            {
                 // tell the timer
                //received=-1;
                //pthread_join(timer, NULL);
                return 0;
            }
            
            else 
            {
                cerr << "Connection closed by peer." << endl;
                // tell the timer
                //received=-1;
                //pthread_join(timer, NULL);
                return 0;
            }
            
	}
        // tell the timer
        //received=1;
        //pthread_join(timer, NULL);
        
	return 1;
}

/*************************************************************************/
// main getRequest function
/*************************************************************************/
extern int dataCheckIn(dataInOut *d, SOCKET sd)
{

        ////////DEBUG//////////
        //d->com_type = SINGLECHTX;
        //d->ndatabytes = 64;
        //return(1);
        ////////////////////////////

        //////////////////////////////////////////////////////////
	// get the com_type
	//printf("waiting for com_type...\n");

        unsigned char* buf_in = new unsigned char[CODEW_SIZE];
	if (!recBufferIn(CODEW_SIZE, buf_in, sd)) {
            delete[] buf_in;
            return 0;
        }
	d->com_type = (unsigned char)*(buf_in);
	//printf("com_type: %d \n", (int)d->com_type);
        delete[] buf_in;

        //////////////////////////////////////////////////////////
	// try to get num data bytes
	//printf("waiting for num databytes...\n");
        buf_in = new unsigned char[NDATABYTES_SIZE];
	if (!recBufferIn(NDATABYTES_SIZE, buf_in, sd)){
            delete[] buf_in;
            return 0;
        }
	d->ndatabytes = (int)*((int*)buf_in);
        delete[] buf_in;
	//printf("databytes: %d \n", m_data.ndatabytes);
        
        //////////////////////////////////////////////////////////
        // if num data bytes > 0, get data...
	if (d->ndatabytes>0)
        {
		//printf("receiving databytes...\n");
                buf_in = new unsigned char[d->ndatabytes];
		if ( !recBufferIn(d->ndatabytes, buf_in, sd) ){
                    delete[] buf_in;
                    return 0;
                }
                
                // alloc space for data
                //d->data = new unsigned char[d->ndatabytes];   //COMMENTATO PER DEBUG
                // save data
		for (int i=0; i<d->ndatabytes; i++) d->data[i] = buf_in[i];
                delete[] buf_in;
	}

        /////////////////////////////////////////////////////////////////
	// get checksum bytes
	//printf("waiting for checksum...\n");
        buf_in = new unsigned char[CHECKSUM_SIZE];
	if ( !recBufferIn(CHECKSUM_SIZE, buf_in, sd) ) {
            delete[] buf_in;
            return 0;
        }
	d->checksum = (short)*((short*)buf_in);
        delete[] buf_in;

// checksum on incoming data
	if (!checkSum(d))
        {
		cerr<<"received data are broken, aborted. \n";
                return 0;
	}

	return 1;
}



/*************************************************************************/
// ShutdownConnection
// Gracefully shuts the connection sock down.  Returns true if we're
// successful, false otherwise.
// return 1 for OK
/*************************************************************************/
extern bool ShutdownConnection(SOCKET sock)
{
    // check if connected
    if (sock==INIT_SOCK) return true; // theres nothing to close
    
    // Disallow any further data sends.  This will tell the other side
    // that we want to go away now.
    if (shutdown(sock, SD_SEND) == SOCKET_ERROR) 
    {
        cout<<"shutdown reported scket err...maybe already closed.\n";
        return false;
    }

    // Receive any extra data still sitting on the socket.  After all
    // data is received, this call will block until the remote host
    // acknowledges the TCP control packet sent by the shutdown above.
    // Then we'll get a 0 back from recv, signalling that the remote
    // host has closed its side of the connection.
    unsigned char* buf_in = new unsigned char[64]; 
    while (1) 
    {
        int nNewBytes = recv(sock, (char*)buf_in, 64, 0);
        if (nNewBytes == SOCKET_ERROR) 
        {
            delete[] buf_in;
            return false;
        }
        else if (nNewBytes != 0) 
        {
            cerr << endl << "FYI, received " << nNewBytes <<" unexpected bytes during shutdown." << endl;
        }
        else 
        {
            // Okay, we're done!
            break;
        }
    }

    // Close the socket.
    if (closesocket(sock) == SOCKET_ERROR) 
    {
        delete[] buf_in;
        return false;
    }

    delete[] buf_in;
    return true;
}


/*************************************************************************/
// calc checkSum bytes
/*************************************************************************/
extern short	 calcSum(dataInOut *d)
{
	short sum=0;
	for (int i=0; i<d->ndatabytes; i++) sum += (short)d->data[i];
	//cout<<"calcSum: "<<sum<<endl;

	return sum;
}


/*************************************************************************/
// checkSum routine
// return 1 for OK
/*************************************************************************/
extern short	checkSum(dataInOut *d)
{
	
	// sum buffer
	short sum=0;
	for (int i=0; i<d->ndatabytes; i++) sum += (short)d->data[i];
	
	//cout<<"checksum...";

	// check
	if (sum!=d->checksum) return 0;
		
	return 1;
}


////////////////////////////////////////////////////
// thread for connections timeouts...
////////////////////////////////////////////////////
void      *timerThread(void *condition)
{
    //cout<<"STARTING timer thread...\n";
    
    Timer t;
    t.start();
    int elapsedTime = 0;
    
    while ( elapsedTime < CONNECTIONS_TIMEOUT )
    {
        elapsedTime = t.getElapsedTimeInSec();    
        if  ( *( (int*)(condition) )!=0  )
        {
      //      cout<<"...stopping timer thread.\n";
			#ifndef _WIN32
            pthread_exit( NULL );
			#endif
        }
    }
    
    cout<<"TIMEOUT Exceeded while waiting for mirror response, abort.\n";
    
    exit(EXIT_FAILURE);
}


////////////////////////////////////////////////////
// create a directory
////////////////////////////////////////////////////

void   createDir(char*  dir){
#ifdef WIN32
	_mkdir(dir);
//#endif
#else
	mkdir (dir,0777);
#endif

}//create dir


////////////////////////////////////////////////



#ifdef WIN32

// rewrite a linux funct for winz
extern void sleep(int secs)
{
	return Sleep(secs*1000);
}

#endif


#ifdef _WIN32
//////////////////// winsock error managing... /////////////
// Statics 
// List of Winsock error constants mapped to an interpretation string.
// Note that this list must remain sorted by the error constants'
// values, because we do a binary search on the list when looking up
// items.
static struct ErrorEntry {

    int nID;
    const char* pcMessage;

    ErrorEntry(int id, const char* pc = 0) : 
    nID(id), 
    pcMessage(pc) { }

    bool operator<(const ErrorEntry& rhs) const
    {
        return nID < rhs.nID;
    }

} gaErrorList[] = {
    ErrorEntry(0,                  "No error"),
    ErrorEntry(WSAEINTR,           "Interrupted system call"),
    ErrorEntry(WSAEBADF,           "Bad file number"),
    ErrorEntry(WSAEACCES,          "Permission denied"),
    ErrorEntry(WSAEFAULT,          "Bad address"),
    ErrorEntry(WSAEINVAL,          "Invalid argument"),
    ErrorEntry(WSAEMFILE,          "Too many open sockets"),
    ErrorEntry(WSAEWOULDBLOCK,     "Operation would block"),
    ErrorEntry(WSAEINPROGRESS,     "Operation now in progress"),
    ErrorEntry(WSAEALREADY,        "Operation already in progress"),
    ErrorEntry(WSAENOTSOCK,        "Socket operation on non-socket"),
    ErrorEntry(WSAEDESTADDRREQ,    "Destination address required"),
    ErrorEntry(WSAEMSGSIZE,        "Message too long"),
    ErrorEntry(WSAEPROTOTYPE,      "Protocol wrong type for socket"),
    ErrorEntry(WSAENOPROTOOPT,     "Bad protocol option"),
    ErrorEntry(WSAEPROTONOSUPPORT, "Protocol not supported"),
    ErrorEntry(WSAESOCKTNOSUPPORT, "Socket type not supported"),
    ErrorEntry(WSAEOPNOTSUPP,      "Operation not supported on socket"),
    ErrorEntry(WSAEPFNOSUPPORT,    "Protocol family not supported"),
    ErrorEntry(WSAEAFNOSUPPORT,    "Address family not supported"),
    ErrorEntry(WSAEADDRINUSE,      "Address already in use"),
    ErrorEntry(WSAEADDRNOTAVAIL,   "Can't assign requested address"),
    ErrorEntry(WSAENETDOWN,        "Network is down"),
    ErrorEntry(WSAENETUNREACH,     "Network is unreachable"),
    ErrorEntry(WSAENETRESET,       "Net connection reset"),
    ErrorEntry(WSAECONNABORTED,    "Software caused connection abort"),
    ErrorEntry(WSAECONNRESET,      "Connection reset by peer"),
    ErrorEntry(WSAENOBUFS,         "No buffer space available"),
    ErrorEntry(WSAEISCONN,         "Socket is already connected"),
    ErrorEntry(WSAENOTCONN,        "Socket is not connected"),
    ErrorEntry(WSAESHUTDOWN,       "Can't send after socket shutdown"),
    ErrorEntry(WSAETOOMANYREFS,    "Too many references, can't splice"),
    ErrorEntry(WSAETIMEDOUT,       "Connection timed out"),
    ErrorEntry(WSAECONNREFUSED,    "Connection refused"),
    ErrorEntry(WSAELOOP,           "Too many levels of symbolic links"),
    ErrorEntry(WSAENAMETOOLONG,    "File name too long"),
    ErrorEntry(WSAEHOSTDOWN,       "Host is down"),
    ErrorEntry(WSAEHOSTUNREACH,    "No route to host"),
    ErrorEntry(WSAENOTEMPTY,       "Directory not empty"),
    ErrorEntry(WSAEPROCLIM,        "Too many processes"),
    ErrorEntry(WSAEUSERS,          "Too many users"),
    ErrorEntry(WSAEDQUOT,          "Disc quota exceeded"),
    ErrorEntry(WSAESTALE,          "Stale NFS file handle"),
    ErrorEntry(WSAEREMOTE,         "Too many levels of remote in path"),
    ErrorEntry(WSASYSNOTREADY,     "Network system is unavailable"),
    ErrorEntry(WSAVERNOTSUPPORTED, "Winsock version out of range"),
    ErrorEntry(WSANOTINITIALISED,  "WSAStartup not yet called"),
    ErrorEntry(WSAEDISCON,         "Graceful shutdown in progress"),
    ErrorEntry(WSAHOST_NOT_FOUND,  "Host not found"),
    ErrorEntry(WSANO_DATA,         "No host data of that type was found")
};

const int kNumMessages = sizeof(gaErrorList) / sizeof(ErrorEntry);



//// WSAGetLastErrorMessage ////////////////////////////////////////////

const char* WSAGetLastErrorMessage(const char* pcMessagePrefix, int nErrorID /* = 0 */)
{
    // Build basic error string
    static char acErrorBuffer[256];
    ostrstream outs(acErrorBuffer, sizeof(acErrorBuffer));
    outs << pcMessagePrefix << ": ";

    // Tack appropriate canned message onto end of supplied message 
    // prefix. Note that we do a binary search here: gaErrorList must be
	// sorted by the error constant's value.
	ErrorEntry* pEnd = gaErrorList + kNumMessages;
    ErrorEntry Target(nErrorID ? nErrorID : WSAGetLastError());
    ErrorEntry* it = lower_bound(gaErrorList, pEnd, Target);
    if ((it != pEnd) && (it->nID == Target.nID)) {
        outs << it->pcMessage;
    }
    else {
        // Didn't find error in list, so make up a generic one
        outs << "unknown error";
    }
    outs << " (" << Target.nID << ")";

    // Finish error message off and return it.
    outs << ends;
    acErrorBuffer[sizeof(acErrorBuffer) - 1] = '\0';

    return acErrorBuffer;
}

#else

/////////////////////////////////////////////////////////////////////////
// ok try to rewrite this useless funct... for linux
const char* WSAGetLastErrorMessage(const char* pcMessagePrefix, int nErrorID /* = 0 */)
{
	
    // Build basic error string
    static char acErrorBuffer[256];
    ostrstream outs(acErrorBuffer, sizeof(acErrorBuffer));
    outs << pcMessagePrefix << ": ";
    outs << "unknown error";
    acErrorBuffer[sizeof(acErrorBuffer) - 1] = '\0';

    return acErrorBuffer;
}

#endif

