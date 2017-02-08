#include "SlaveWorker.h"
using namespace metis_plat; 
#include "woo/log.h"
#include <string.h>


SlaveTrainer SlaveWorker::m_slaveTrainer; 
bool SlaveWorker::m_bIsUpdating = false;
double SlaveWorker::m_dAvgLoss = 0.011111;
CalcGradParamsT SlaveWorker::m_calcGradParamsT; 


SlaveWorker::SlaveWorker()
{
}


SlaveWorker::~SlaveWorker()
{
}


void SlaveWorker::WorkCore(Json::Value& jsonReq, Json::Value& jsonResp)
{
	jsonResp.clear();
	try 
	{
		string str_cmd = jsonReq["cmd"].asString();
		if(str_cmd == "detect")
			Work_Detect(jsonResp);
		else if(str_cmd == "get_patt_cnt")
			Work_GetPattCnt(jsonResp);
		else if(str_cmd == "shuffle_patt")
			Work_ShufflePatt(jsonResp);
		else if(str_cmd == "calc_grad")
			Work_CalcGrad(jsonReq["body"], jsonResp); 
		else if(str_cmd == "release_weight")
			Work_ReleaseWeight(jsonResp);
		else if(str_cmd == "release")
			Work_Release(jsonResp);
		else if(str_cmd == "reset")
			Work_Reset(jsonResp);
		else if(str_cmd == "is_updating")
			Work_IsUpdating(jsonResp);
		else if(str_cmd == "patts_info")
			Work_PattsInfo(jsonResp);
		else
		{
			jsonResp["ret"] = _METIS_PLAT_UNSUPPORT_CMD; 
			jsonResp["msg"] = "unsupport cmd";
			LOG_ERROR("CMD [%s] is Unsupported!", str_cmd.c_str()); 
		}
	}
	catch(...)
	{
		jsonResp["ret"] = _METIS_PLAT_EXCEPTION_OCCUR;
		jsonResp["msg"] = "exception occur";
		LOG_ERROR("Exception occur");
	}
}


void SlaveWorker::WorkCore_bitStream(const char* bitStream, const int32_t nLen, Json::Value& jsonResp)
{
	char scmd[_CMD_MAX_SIZE];
	scmd[0] = '\0'; 
	int32_t off = 0;
	while(off < nLen && off < _CMD_MAX_SIZE)
	{
		if(bitStream[off] == '@')
		{
			memcpy(scmd, bitStream, off); 
			scmd[off] = '\0'; 
			off++; 
			break;
		}
		off++; 
	}
	
	jsonResp.clear(); 
	if(strlen(scmd) > 0)
	{
		if(strcmp(scmd, "push_patt") == 0)
			Work_PushPatt(bitStream + off, nLen - off, jsonResp); 
		else if(strcmp(scmd, "push_patt_string") == 0)
			Work_PushPattString(bitStream + off, jsonResp); 
		else if(strcmp(scmd, "set_weight") == 0)
			Work_SetWeight(bitStream + off, nLen - off, jsonResp); 
		else
		{
			jsonResp["ret"] = _METIS_PLAT_UNSUPPORT_CMD; 
			jsonResp["msg"] = "unsupport cmd";
			LOG_ERROR("CMD [%s] is Unsupported!", scmd); 
		}
	}
	else
	{
		jsonResp["ret"] = _METIS_PLAT_INCORRECT_BITSTREAM;
		jsonResp["msg"] = "incorrect bit stream!"; 
		LOG_ERROR("Incorrect bit-stream!");
	}
}


void SlaveWorker::WorkCore_GetBatchGrad(char* bitStream_Grad, int32_t& nLen)
{
	bool is_updating; 

	m_rwLock.WrLock(); 
	is_updating = m_bIsUpdating;
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		strcpy(bitStream_Grad, "updating##");
		nLen = strlen("updating##");
	}
	else
	{
		nLen = 0; 
		memcpy(bitStream_Grad + nLen, &m_dAvgLoss, sizeof(double)); 
		nLen += sizeof(double); 
		nLen += m_slaveTrainer.GetGrad_asStream(bitStream_Grad + nLen); 
	}
}



// 处理"detect"请求
void SlaveWorker::Work_Detect(Json::Value& jsonResp)
{
	jsonResp["ret"] = _METIS_PLAT_SUCCESS;
	jsonResp["msg"] = "ok";
	LOG_INFO("cmd[detect], success"); 
}


// 处理"get_patt_cnt"请求
void SlaveWorker::Work_GetPattCnt(Json::Value& jsonResp)
{
	jsonResp["ret"] = _METIS_PLAT_SUCCESS;
	jsonResp["msg"] = "ok";
	m_rwLock.RdLock(); 
	jsonResp["patts"] = m_slaveTrainer.PattCnt(); 
	m_rwLock.Unlock(); 
	LOG_INFO("cmd[get_patt_cnt], success"); 
}


// 处理"shuffle_patt"请求
void SlaveWorker::Work_ShufflePatt(Json::Value& jsonResp)
{
	bool is_updating; 
	
	m_rwLock.WrLock();
	is_updating = m_bIsUpdating;
	if(!is_updating)
		m_slaveTrainer.PattShuffle(); 
	m_rwLock.Unlock();

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING; 
		jsonResp["msg"] = "model updating";
		LOG_DEBUG("cmd[shuffle_patt], model updating"); 
	}
	else
	{
		jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
		jsonResp["msg"] = "ok"; 
		LOG_INFO("cmd[shuffle_patt], success"); 
	}
}

// 处理"calc_grad"请求
void SlaveWorker::Work_CalcGrad(Json::Value& jsonReqBody, Json::Value& jsonResp)
{
	bool is_updating; 

	m_rwLock.RdLock();
	is_updating = m_bIsUpdating; 
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING; 
		jsonResp["msg"] = "model updating";
		LOG_DEBUG("cmd[calc_grad], model updating"); 
	}
	else
	{
		m_rwLock.WrLock();
		m_calcGradParamsT.batch_size = jsonReqBody["batch_size"].asInt(); 
		m_calcGradParamsT.hidden_act = ActConv::ActType(jsonReqBody["hidden_act"].asString().c_str()); 
		m_rwLock.Unlock(); 

		pthread_t th_id;
		int32_t ret = pthread_create(&th_id, NULL, Thread_CalcBatchGrad, NULL); 
		if(ret == 0) 
		{
			jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
			jsonResp["msg"] = "ok";
			LOG_INFO("cmd[calc_grad], calculation has been started successfully"); 
		}
		else
		{
			jsonResp["ret"] = _METIS_PLAT_CALCULATIONSTART_FAIL; 
			jsonResp["msg"] = "start calculation fail"; 
			LOG_DEBUG("cmd[calc_grad], failed to start calculation"); 
		}
	}
}


// 处理"release_weight"请求
void SlaveWorker::Work_ReleaseWeight(Json::Value& jsonResp)
{
	bool is_updating; 

	m_rwLock.WrLock();
	is_updating = m_bIsUpdating; 
	if(!is_updating)
		m_slaveTrainer.ReleaseWeight(); 
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING;
		jsonResp["msg"] = "model updating";
		LOG_DEBUG("[release_weight], model updating"); 
	}
	else
	{
		jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
		jsonResp["msg"] = "ok"; 
		LOG_INFO("[release_weight], weight in slave trainer has been released successfully"); 
	}
}


// 处理"release"请求
void SlaveWorker::Work_Release(Json::Value& jsonResp)
{
	bool is_updating; 

	m_rwLock.WrLock();
	is_updating = m_bIsUpdating; 
	if(!is_updating)
	{
		m_slaveTrainer.ReleaseWeight(); 
		m_slaveTrainer.ReleasePatts(); 
	}
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING;
		jsonResp["msg"] = "model updating";
		LOG_DEBUG("[release], model updating"); 
	}
	else
	{
		jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
		jsonResp["msg"] = "ok"; 
		LOG_INFO("[release], slave trainer has been released successfully"); 
	}
}


// 处理"reset"请求
void SlaveWorker::Work_Reset(Json::Value& jsonResp)
{
	if(m_bIsUpdating)
	{
		m_slaveTrainer.SetCancelFlag(true); 
		m_slaveTrainer.ReleaseWeight(); 
		m_slaveTrainer.ReleasePatts(); 
		m_bIsUpdating = false; 
	}
	jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
	jsonResp["msg"] = "ok"; 
	LOG_INFO("cmd[reset], slave trainer has been reset successfully"); 
}


// 处理"push_patt_string"
void SlaveWorker::Work_PushPattString(const char* sPattStr, Json::Value& jsonResp)
{
	bool is_updating, flag; 

	m_rwLock.WrLock(); 
	is_updating = m_bIsUpdating;
	if(!is_updating)
		flag = m_slaveTrainer.PushPatt_inString(sPattStr); 
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING; 
		jsonResp["msg"] = "model updating";
		LOG_DEBUG("cmd[push_patt_string], model updating"); 
	}
	else
	{
		if(flag)
		{
			jsonResp["ret"] = _METIS_PLAT_SUCCESS;
			jsonResp["msg"] = "ok";
			LOG_INFO("cmd[push_patt_string], receive pattern from master successfully"); 
		}
		else
		{
			jsonResp["ret"] = _METIS_PLAT_PUSHPATT_FAIL;
			jsonResp["msg"] = "push fail"; 
			LOG_ERROR("cmd[push_patt_string], pattern pushing fail"); 
		}
	}
}


// 处理"push_patt"请求
void SlaveWorker::Work_PushPatt(const char* bitStream_Patt, const int32_t nLen, Json::Value& jsonResp)
{
	bool is_updating, flag; 

	m_rwLock.WrLock(); 
	is_updating = m_bIsUpdating;
	if(!is_updating)
		flag = m_slaveTrainer.PushPatt_inStream(bitStream_Patt, nLen); 
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING; 
		jsonResp["msg"] = "model updating";
		LOG_DEBUG("cmd[push_patt], model updating"); 
	}
	else
	{
		if(flag)
		{
			jsonResp["ret"] = _METIS_PLAT_SUCCESS;
			jsonResp["msg"] = "ok";
			LOG_DEBUG("cmd[push_patt], receive pattern from master successfully"); 
		}
		else
		{
			jsonResp["ret"] = _METIS_PLAT_PUSHPATT_FAIL;
			jsonResp["msg"] = "push fail"; 
			LOG_ERROR("cmd[push_patt], pattern pushing fail"); 
		}
	}
}


// 处理"set_weight"请求
void SlaveWorker::Work_SetWeight(const char* bitStream_Weight, const int32_t nLen, Json::Value& jsonResp)
{
	bool is_updating, flag; 

	m_rwLock.WrLock(); 
	is_updating = m_bIsUpdating;
	if(!is_updating)
		flag = m_slaveTrainer.SetWeight_inStream(bitStream_Weight, nLen); 
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING; 
		jsonResp["msg"] = "model updating";
		LOG_DEBUG("cmd[set_weight], model updating");
	}
	else
	{
		if(flag)
		{
			jsonResp["ret"] = _METIS_PLAT_SUCCESS;
			jsonResp["msg"] = "ok";
			LOG_INFO("cmd[set_weight], set weight successfully"); 
		}
		else
		{
			jsonResp["ret"] = _METIS_PLAT_SETWEIGHT_FAIL;
			jsonResp["msg"] = "set fail"; 
			LOG_ERROR("cmd[set_weight], failed to set weight"); 
		}
	}
}


// 处理"is_updating"请求
void SlaveWorker::Work_IsUpdating(Json::Value& jsonResp)
{
	bool is_updating; 

	m_rwLock.WrLock();
	is_updating = m_bIsUpdating; 
	m_rwLock.Unlock(); 

	jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
	jsonResp["msg"] = "ok"; 
	if(is_updating)
		jsonResp["updating"] = true; 
	else
		jsonResp["updating"] = false; 
}


// 处理"patts_info"请求
void SlaveWorker::Work_PattsInfo(Json::Value& jsonResp)
{
	PattsInfoT patts_info_t = m_slaveTrainer.GetPattsInfo(); 

	jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
	jsonResp["msg"] = "ok"; 
	jsonResp["x_dims"] = patts_info_t.x_dims; 
	jsonResp["y_dims"] = patts_info_t.y_dims; 
	jsonResp["patts"] = patts_info_t.patts; 
}


void* SlaveWorker::Thread_CalcBatchGrad(void* pParam)
{
	pthread_detach(pthread_self());

	m_rwLock.WrLock();
	m_bIsUpdating = true;
	m_rwLock.Unlock();

	m_slaveTrainer.SetCancelFlag(false); 
	m_slaveTrainer.CalcBatchGrad(m_dAvgLoss, m_calcGradParamsT.batch_size, m_calcGradParamsT.hidden_act); 
	
	m_rwLock.WrLock();
	m_bIsUpdating = false;
	m_rwLock.Unlock();

	pthread_exit(0);
	return NULL; 
}




