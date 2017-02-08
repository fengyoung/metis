#include <iostream>
using namespace std;
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <ext/hash_set>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <limits.h>
#include "woo/log.h"
#include "woo/tcpserver.h"

#include "SlaveWorker.h"
#include "JsonTransf.h"
#include "ServConf.h"
using namespace metis_plat; 

static SlaveWorker g_worker; 
static ServConf g_servconf; 


int req_handle(void *handle_data, char *input, uint32_t input_len, char *output, uint32_t *output_len, char *msg, size_t msg_size) 
{
	woo::binary_head_t* req_head = (woo::binary_head_t*)input;
	char* req_body = input + sizeof(woo::binary_head_t);
	woo::binary_head_t* resp_head = (woo::binary_head_t*)output;
	char* resp_body = output + sizeof(woo::binary_head_t);
	input[input_len] = '\0';

	int32_t off = 0; 
	int32_t req_body_len = (int32_t)input_len - (int32_t)sizeof(woo::binary_head_t); 
	while(off < req_body_len)
	{
		if(req_body[off] == '@')
			break; 
		off++; 	
	}

	Json::Value json_req, json_resp; 
	bool as_json = true; 	
	memset(resp_head, 0, sizeof(woo::binary_head_t));
	
	if(off == req_body_len)
	{
		json_resp["ret"] = _METIS_PLAT_PARSEREQ_HEADER_FAIL; 
		json_resp["msg"] = "parse request header fail"; 
		LOG_ERROR("Failed to parse request header"); 
	}
	else
	{
		if(strncmp(req_body, "METIS_CMD@", strlen("METIS_CMD@")) == 0)
		{
			if(JsonTransf::StringToJsonCpp(json_req, req_body + off + 1))
			{
				g_worker.WorkCore(json_req, json_resp); 
			}
			else
			{
				json_resp["ret"] = _METIS_PLAT_PARSEREQ_FAIL; 
				json_resp["msg"] = "parse request json fail"; 
				LOG_ERROR("Failed to parse request json"); 
			}
		}
		else if(strncmp(req_body, "BITSTREAM@", strlen("BITSTREAM@")) == 0)
		{
			g_worker.WorkCore_bitStream(req_body + off + 1, req_body_len - off - 1, json_resp); 				
		}
		else if(strncmp(req_body, "GETBATCHGRAD@", strlen("GETBATCHGRAD@")) == 0)
		{
			int32_t len; 
			g_worker.WorkCore_GetBatchGrad(resp_body, len); 
			resp_head->body_len = len; 
			as_json = false; 	
		}
		else
		{
			json_resp["ret"] = _METIS_PLAT_PARSEREQ_HEADER_FAIL; 
			json_resp["msg"] = "parse request header fail"; 
			LOG_ERROR("Failed to parse request header"); 
		}
	}

	resp_head->log_id = req_head->log_id;
	if(as_json)
	{
		string str_resp = JsonTransf::JsonCppToString(json_resp); 
		strcpy(resp_body, str_resp.c_str()); 
		resp_head->body_len = str_resp.length(); 
	}

	*output_len = sizeof(woo::binary_head_t) + resp_head->body_len;
	output[*output_len] = '\0';

	return 0; 
}



int main(int argc, char** argv)
{
	if(argc != 3)
	{
		cout<<"usage: "<<argv[0]<<" <port> <config_file>"<<endl; 
		return -1; 
	}

	int32_t port; 
	sscanf(argv[1], "%d", &port); 
	if(!g_servconf.Read(argv[2], port))
	{
		cout<<"failed to read config file "<<argv[2]<<endl; 
		return -2; 	
	}

	woo::open_log(g_servconf.LogFile(false).c_str(), 1024*100, g_servconf.m_strLogLev.c_str(), g_servconf.m_strLogLev.c_str()); 
	g_worker.Init(g_servconf); 

	woo::tcp_server_t *server = woo::tcp_server_create();
	int32_t ret = woo::tcp_server_open(server, g_servconf.m_strHost.c_str(), g_servconf.m_nPort, woo::binary_recv, req_handle, NULL, 
			g_servconf.m_nThreadNum, g_servconf.m_bLongConn, g_servconf.m_nRecvBufSize, g_servconf.m_nSendBufSize, 
			g_servconf.m_nRecvTimeout * 1000, g_servconf.m_nSendTimeout * 1000); 
	if(ret < 0)
	{
		cout<<"Server error!"<<endl; 
		return -1;	
	}	
	woo::tcp_server_run(server);
	woo::tcp_server_wait(server);
	woo::tcp_server_destroy(server);

	return 0; 
}


