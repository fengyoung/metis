#ifndef _METIS_PLATFORM_WORKER_H 
#define _METIS_PLATFORM_WORKER_H 

#include <string>
using namespace std;
#include <stdint.h>
#include "ServConf.h"
#include "ThreadRWLock.h"
#include "JsonTransf.h"

#define _RESP_BUF_SIZE  (1024*1024*8)

#define _METIS_PLAT_SUCCESS		0
#define _METIS_PLAT_ISUPDATING  1
#define _METIS_PLAT_PARSEREQ_FAIL	-1

#define _METIS_PLAT_EXCEPTION_OCCUR	-11
#define _METIS_PLAT_UNSUPPORT_CMD	-12

#define _METIS_PLAT_PUSHPATT_FAIL	-101
#define _METIS_PLAT_SETMODEL_FAIL	-102
#define _METIS_PLAT_GETMOEL_FAIL	-103
#define _METIS_PLAT_UPDATESTART_FAIL	-104


namespace metis_plat
{

class Worker 
{
public: 
	Worker(); 
	virtual ~Worker(); 

	void Init(ServConf& servConf); 
	
	static bool Send(const char* sHost, const int32_t nPort, const char* sCmd, Json::Value& jsonReqBody, Json::Value& jsonResp, 
			const int32_t nLogId = 2016, const int32_t nSendTimeoutMs = 1000, const int32_t nRecvTimeoutMs = 1000); 
	static bool Send(const char* sHostPort, const char* sCmd, Json::Value& jsonReqBody, Json::Value& jsonResp, 
			const int32_t nLogId = 2016, const int32_t nSendTimeoutMs = 1000, const int32_t nRecvTimeoutMs = 1000); 

	virtual void WorkCore(Json::Value& jsonReq, Json::Value& jsonResp) = 0; 

protected:
	ServConf m_servConf; 
	static ThreadRWLock m_rwLock; 
};

}

#endif /* _METIS_PLATFORM_WORKER_H */

