#include "Worker.h"
#include "Ether.h"
using namespace metis_plat; 
#include <iostream>
using namespace std; 
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "woo/binaryclient.h"
#include "woo/log.h"

#include "StringArray.h"
using namespace metis_uti; 


ThreadRWLock Worker::m_rwLock; 


Worker::Worker()
{
}


Worker::~Worker()
{
}


void Worker::Init(ServConf& servConf)
{
	m_servConf = servConf; 
}


bool Worker::Send(const char* sHostPort, const char* sCmd, Json::Value& jsonReqBody, Json::Value& jsonResp, 
		const int32_t nLogId, const int32_t nSendTimeoutMs, const int32_t nRecvTimeoutMs) 
{
	string host; 
	int32_t port;
	Ether::StringToHostPort(host, port, sHostPort);

	Json::Value json_req; 
	json_req["cmd"] = sCmd; 
	json_req["body"] = jsonReqBody; 
	string str_req = JsonTransf::JsonCppToString(json_req); 
	uint32_t send_len = str_req.length(); 
	char* s_resp = new char[_RESP_BUF_SIZE];
	uint32_t recv_len; 	

	woo::binary_client_t *cli = woo::binary_client_create(host.c_str(), port, nSendTimeoutMs * 1000, nRecvTimeoutMs * 1000);  
	ssize_t ret = woo::binary_client_talk(cli, str_req.c_str(), send_len, s_resp, &recv_len, _RESP_BUF_SIZE, nLogId); 
	binary_client_destroy(cli); 

	if(ret)
	{
		delete s_resp;  
		return false; 
	}	

	s_resp[recv_len] = '\0'; 
	if(!JsonTransf::StringToJsonCpp(jsonResp, s_resp))
	{
		delete s_resp;  
		return false; 
	}

	delete s_resp;  
	return true; 
}


