/***********************************************************************

  mirrorDriver.h

  multiplatform mirrorDriver class

  copyright (c) ADAPTICA 2009

***********************************************************************/


#ifndef _MIRROR_DRIVER_H
#define _MIRROR_DRIVER_H


//#include "../utils/ws-util.h"
#include "../Header/ws-util.h"
#include "../Header/serverSocketLnx.h"
#include <string.h>
#include <math.h>


/*****************************************/
// client class definition
/*****************************************/
class mirrorDriver
{

public:
        /****************************************/
	// constructor
        /****************************************/
	mirrorDriver(void);
	
	/****************************************/
        // destructor
        /****************************************/
	~mirrorDriver(void);
	
        /****************************************/
	// connect to mirror
	//
	// return: true if success
	//
        /****************************************/
	int connectToMirror();
        
        /****************************************/
	// disconnect from mirror
	//
	// return: true if success
	//
        /****************************************/
	int disconnect();
        
        /****************************************/
	// return: true if correctly connected
	//
	// return: true if success
	//
        /****************************************/
        int isConnected(void);
        

       /*********************************************************************/
       // MAIN FUNCTIONS: requests managed by mirachServer
       /*********************************************************************/ 
        
        /****************************************/
	//
	// set full-channels data
	//
	// arg:
	// return: true if success
	//
        /****************************************/
	int		setChannels(float* mdata);


        /****************************************/
		//
		// set single data for mirror channel
		//
		// args: 1-value, 2-channel number 
		// return: true if success
		//
        /****************************************/
		int		setSingleChannel(float sdata, int chnum);

        /****************************************/
		//
		// return: server IP
		//
        /****************************************/
		char*           getServerIPAddress(void);
        
        /****************************************/ 
		//
		// reset mirror channels to default vals
		//
		// return: true if success
		//
        /****************************************/
		int		resetAllChans(void);


        /****************************************/
		//
		// reset a single channel to default val
		//
		// arg: channel number
		// return: true if success
		//
        /****************************************/
		int		resetChan(int chnum);


        /****************************************/
		//
		// get mirror status for all channels
		//
		// arg: pointer to float array of size = 
		//		number of channels
		// return: true if success
		//
        /****************************************/
		int		getChannelsStatus(float* ret);

        /****************************************/
		//
		// get mirror num of channels
		//
		// return: number of mirror channels
		//
        /****************************************/
		int		getNumMirrorChannels(void);


        /****************************************/
		//
		// get mirror status for a single channel
		//
		//	arg: channel number
		//
		//	return: mirror channel current value
		//
        /****************************************/
		float   getSingleChannelStatus(int chnum);


        /****************************************/
		//
		// return: current driver version
		//
        /****************************************/
		float	getDriverVersion(void);
        

        /*******************************************************************/
		//
        //  SUPERVISOR-MANAGED SERVICES
        //  these functions are sent via the service socket to the server 
        //  supervisor thread (mirachSrvrSupervisor)
		//
        /*******************************************************************/
        
        /****************************************/
		//
        // config server IP address
		//
		// return: true if success
		//
        /****************************************/
        int             configIPAddress(char* ip_addr);
        
        /****************************************/
        // reboot server main thread
		//
		// return: true if success
		//
        /****************************************/
        int             resetMirrorDriver(void);
        
        /****************************************/
        // mirror firmware update
		//
        // arg: pointer to the bin file
		//
		// return: true if success
        /****************************************/
        int             updateServerFirmware(char* updatefile);
        
       
private:

		/****************************************/
        // reset connection with mirror
		//
		// return: true if success
		//
        /****************************************/
        int             resetConnection(void);


		/****************************************/
        // check supervisor and server ID,send ID
		//
		// return: true if success
		//
		/****************************************/
        int             handshakes(void);
		
		/****************************************/
		// receive response buffer
		//
		// return: true if success
		//
		/****************************************/
		int		recResponse(unsigned char com_type, SOCKET sd);


		/****************************************/
		// get mirror number of channels 
		// from mirachServer & stores in the internal 
		// datafield
		//
		// return: true if success
		//
		/****************************************/
		int		getNumChannels(void);


		/****************************************/
		// read IP addr. from config file
		//
		// return: true if success
		//
		/****************************************/
		int		readConfigFile(char *configFile);
	        
        
		////////////////////////////////////////////////////////////////////////////////
		// PRIVATE DATAFIELDs
	        
		// connection sockets
		SOCKET          mainSock,serviceSock;

		// connection address
		char*		serverIPAddress;
	        
		// ports
		int     mainPort,servicePort;

		// connection flag
		int		connected;

		// driver version
		float		driverVers;

		// number of mirror channels
		short		chnum;

		// I/O data
		dataInOut	m_data;
	        
		// a tag for debug purposes
		char*       class_id;

};


#endif

