#ifndef _METIS_PLATFORM_SERV_CONF_H 
#define _METIS_PLATFORM_SERV_CONF_H 

#include <string>
#include <set>
using namespace std; 
#include <stdint.h>

namespace metis_plat
{

class ServConf
{
public:
	ServConf(); 
	ServConf(const ServConf& servConf); 
	virtual ~ServConf(); 

	ServConf& operator = (const ServConf& servConf); 
	
	bool Read(const char* sConfigFile, const int32_t nPort = 0); 
	string LogFile(const bool bIsMaster); 

public:
	string m_strHost; 
	int32_t m_nPort; 
	
	bool m_bLongConn;
	int32_t m_nThreadNum;
	int32_t m_nRecvBufSize;
	int32_t m_nSendBufSize;
	int32_t m_nRecvTimeout;	// recvice timeout (ms) 
	int32_t m_nSendTimeout;	// send timeout (ms) 

	string m_strLogPath; 
	string m_strLogLev;
};

}

#endif /* _METIS_PLATFORM_SERV_CONF_H */ 


