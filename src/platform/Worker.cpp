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


bool Worker::SendCmd(const char* sHostPort, const char* sCmd, Json::Value& jsonReqBody, Json::Value& jsonResp, 
		const int32_t nLogId, const int32_t nSendTimeoutMs, const int32_t nRecvTimeoutMs) 
{
	string host; 
	int32_t port;
	Ether::StringToHostPort(host, port, sHostPort);

	Json::Value json_req; 
	json_req["cmd"] = sCmd; 
	json_req["body"] = jsonReqBody; 
	string str_req = "METIS_CMD@" + JsonTransf::JsonCppToString(json_req); 
	uint32_t send_len = str_req.length(); 
	char* s_resp = new char[_RESP_BUF_SIZE];
	uint32_t recv_len; 	

	woo::binary_client_t *cli = woo::binary_client_create(host.c_str(), port, nSendTimeoutMs * 1000, nRecvTimeoutMs * 1000);  
	ssize_t ret = woo::binary_client_talk(cli, str_req.c_str(), send_len, s_resp, &recv_len, _RESP_BUF_SIZE, nLogId); 
	if(ret)
	{
		ret = woo::binary_client_talk(cli, str_req.c_str(), send_len, s_resp, &recv_len, _RESP_BUF_SIZE, nLogId); 
		if(ret)
		{
			binary_client_destroy(cli); 
			delete s_resp;  
			return false; 
		}
	}	
	binary_client_destroy(cli); 

	s_resp[recv_len] = '\0'; 
	if(!JsonTransf::StringToJsonCpp(jsonResp, s_resp))
	{
		delete s_resp;  
		return false; 
	}
	delete s_resp;  
	return true; 
}


/*
bool Worker::SendBinaryPatt(const char* sHostPort, Pattern* pPatt, Json::Value& jsonResp, 
		const int32_t nLogId, const int32_t nSendTimeoutMs, const int32_t nRecvTimeoutMs)
{
	string host; 
	int32_t port;
	Ether::StringToHostPort(host, port, sHostPort);

	int32_t len = (pPatt->m_nXCnt + pPatt->m_nYCnt) * sizeof(double) + 2 * sizeof(int32_t) + strlen("METIS_BINARY_PATT@"); 
	char* stream = new char[len * 2]; 
	char* s_resp = new char[_RESP_BUF_SIZE];
	uint32_t recv_len; 	

	strcpy(stream, "METIS_BINARY_PATT@"); 
	len = strlen("METIS_BINARY_PATT@"); 
	len += pPatt->ToStream(stream + len); 

	woo::binary_client_t *cli = woo::binary_client_create(host.c_str(), port, nSendTimeoutMs * 1000, nRecvTimeoutMs * 1000);  
	ssize_t ret = woo::binary_client_talk(cli, stream, len, s_resp, &recv_len, _RESP_BUF_SIZE, nLogId); 
	binary_client_destroy(cli); 

	if(ret)
	{
		delete stream; 
		delete s_resp;  
		return false; 
	}	

	s_resp[recv_len] = '\0'; 
	if(!JsonTransf::StringToJsonCpp(jsonResp, s_resp))
	{
		delete stream; 
		delete s_resp;  
		return false; 
	}

	delete stream;
	delete s_resp; 
	return true; 
}
*/


// "GETBATCHGRAD@"
bool Worker::GetBatchGrad(const char* sHostPort, char* bitStream_Grad, int32_t& nLen, const int32_t nBuffSize,  
			const int32_t nLogId, const int32_t nSendTimeoutMs, const int32_t nRecvTimeoutMs)
{
	string host; 
	int32_t port;
	Ether::StringToHostPort(host, port, sHostPort);

	char stream[] = "GETBATCHGRAD@";
	int32_t len = strlen(stream);  
	uint32_t recv_len; 	

	woo::binary_client_t *cli = woo::binary_client_create(host.c_str(), port, nSendTimeoutMs * 1000, nRecvTimeoutMs * 1000);  
	ssize_t ret = woo::binary_client_talk(cli, stream, len, bitStream_Grad, &recv_len, nBuffSize, nLogId); 
	binary_client_destroy(cli); 

	if(ret)
		return false; 

	bitStream_Grad[recv_len] = '\0';
	nLen = (int32_t)recv_len;  
	return true; 
}


// "BITSTREAM@[CMD]@[body...]"
bool Worker::SendBitStream(const char* sHostPort, const char* sCmd, const char* bitStreamBody, const int32_t nBodyLen, Json::Value& jsonResp, 
			const int32_t nLogId, const int32_t nSendTimeoutMs, const int32_t nRecvTimeoutMs)
{
	string host; 
	int32_t port;
	Ether::StringToHostPort(host, port, sHostPort);

	int32_t len = 0; 
	char* stream = new char[nBodyLen + _CMD_MAX_SIZE + 64];
	char* s_resp = new char[_RESP_BUF_SIZE];
	uint32_t recv_len; 	

	strcpy(stream + len, "BITSTREAM@");
	len += strlen("BITSTREAM@");

	memcpy(stream + len, sCmd, strlen(sCmd)); 
	len += strlen(sCmd); 
	stream[len] = '@'; 
	len += 1; 

	memcpy(stream + len, bitStreamBody, nBodyLen); 
	len += nBodyLen;
	woo::binary_client_t *cli = woo::binary_client_create(host.c_str(), port, nSendTimeoutMs * 1000, nRecvTimeoutMs * 1000);  
	ssize_t ret = woo::binary_client_talk(cli, stream, len, s_resp, &recv_len, _RESP_BUF_SIZE, nLogId); 
	binary_client_destroy(cli); 

	if(ret)
	{
		delete stream; 
		delete s_resp;  
		return false; 
	}	

	s_resp[recv_len] = '\0'; 
	if(!JsonTransf::StringToJsonCpp(jsonResp, s_resp))
	{
		delete stream; 
		delete s_resp;  
		return false; 
	}

	delete stream;
	delete s_resp; 
	return true; 
} 

