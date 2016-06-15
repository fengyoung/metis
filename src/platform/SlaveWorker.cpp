#include "SlaveWorker.h"
using namespace metis_plat; 
#include "woo/log.h"


SlaveTrainer SlaveWorker::m_slaveTrainer; 
bool SlaveWorker::m_bIsUpdating = false;
UpdateModelParamT SlaveWorker::m_umParamT; 


SlaveWorker::SlaveWorker() : Worker()
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
		else if(str_cmd == "push_patt")
			Work_PushPatt(jsonReq["body"], jsonResp);
		else if(str_cmd == "get_patt_cnt")
			Work_GetPattCnt(jsonResp);
		else if(str_cmd == "shuffle_patt")
			Work_ShufflePatt(jsonResp);
		else if(str_cmd == "set_model")
			Work_SetModel(jsonReq["body"], jsonResp);
		else if(str_cmd == "get_model")
			Work_GetModel(jsonResp);
		else if(str_cmd == "once_update_model")
			Work_OnceUpdateModel(jsonReq["body"], jsonResp);
		else if(str_cmd == "loop_update_model")
			Work_LoopUpdateModel(jsonReq["body"], jsonResp);
		else if(str_cmd == "release")
			Work_Release(jsonResp);
		else if(str_cmd == "reset")
			Work_Reset(jsonResp);
		else if(str_cmd == "is_updating")
			Work_IsUpdating(jsonResp);
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


// 处理"detect"请求
void SlaveWorker::Work_Detect(Json::Value& jsonResp)
{
	jsonResp["ret"] = _METIS_PLAT_SUCCESS;
	jsonResp["msg"] = "ok";
	LOG_DEBUG("Receive detection from Master"); 
}


// 处理"push_patt"请求
void SlaveWorker::Work_PushPatt(Json::Value& jsonReqBody, Json::Value& jsonResp)
{
	bool is_updating, flag; 
	
	m_rwLock.WrLock(); 
	is_updating = m_bIsUpdating;
	if(!is_updating)
		flag = m_slaveTrainer.PushPatt(jsonReqBody["patt"].asString().c_str()); 
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING; 
		jsonResp["msg"] = "model updating";
	}
	else
	{
		if(flag)
		{
			jsonResp["ret"] = _METIS_PLAT_SUCCESS;
			jsonResp["msg"] = "ok";
			LOG_DEBUG("Receive pattern from Master"); 
		}
		else
		{
			jsonResp["ret"] = _METIS_PLAT_PUSHPATT_FAIL;
			jsonResp["msg"] = "push fail"; 
			LOG_ERROR("Failed to push pattern | %s", JsonTransf::JsonCppToString(jsonReqBody).c_str()); 
		}
	}
}


// 处理"get_patt_cnt"请求
void SlaveWorker::Work_GetPattCnt(Json::Value& jsonResp)
{
	jsonResp["ret"] = _METIS_PLAT_SUCCESS;
	jsonResp["msg"] = "ok";
	m_rwLock.RdLock(); 
	jsonResp["patts"] = m_slaveTrainer.PattCnt(); 
	m_rwLock.Unlock(); 
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
	}
	else
	{
		jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
		jsonResp["msg"] = "ok"; 
		LOG_DEBUG("Shuffle");
	}
}


// 处理"set_model"请求
void SlaveWorker::Work_SetModel(Json::Value& jsonReqBody, Json::Value& jsonResp)
{
	bool is_updating, flag; 

	m_rwLock.WrLock(); 
	is_updating = m_bIsUpdating;	
	if(!is_updating)
		flag = m_slaveTrainer.SetModel(jsonReqBody["model"].asString().c_str()); 
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING; 
		jsonResp["msg"] = "model updating";
	}
	else
	{
		if(flag)
		{
			jsonResp["ret"] = _METIS_PLAT_SUCCESS;
			jsonResp["msg"] = "ok";
			LOG_DEBUG("Set model"); 
		}
		else
		{
			jsonResp["ret"] = _METIS_PLAT_SETMODEL_FAIL;
			jsonResp["msg"] = "set fail"; 
			LOG_ERROR("Faield to set model | %s", JsonTransf::JsonCppToString(jsonReqBody).c_str()); 
		}
	}
}


// 处理"get_model"请求
void SlaveWorker::Work_GetModel(Json::Value& jsonResp)
{
	string str_model;
	double avg_error; 
	bool is_updating; 

	m_rwLock.RdLock();
	is_updating = m_bIsUpdating; 
	if(!is_updating)
	{
		str_model = m_slaveTrainer.GetModel(); 
		avg_error = m_slaveTrainer.GetAvgError();
	}
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING; 
		jsonResp["msg"] = "model updating";
	}
	else
	{
		if(!str_model.empty())
		{
			jsonResp["ret"] = _METIS_PLAT_SUCCESS;
			jsonResp["msg"] = "ok";
			jsonResp["model"] = str_model; 	
			jsonResp["avg_error"] = avg_error; 	
		}
		else
		{
			jsonResp["ret"] = _METIS_PLAT_GETMOEL_FAIL;
			jsonResp["msg"] = "get fail"; 
			LOG_ERROR("Faield to get model"); 
		}
	}
}


// 处理"once_update_model"请求
void SlaveWorker::Work_OnceUpdateModel(Json::Value& jsonReqBody, Json::Value& jsonResp)
{
	bool is_updating; 

	m_rwLock.RdLock();
	is_updating = m_bIsUpdating; 
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING; 
		jsonResp["msg"] = "model updating";
	}
	else
	{
		m_rwLock.WrLock();
		m_umParamT.batch_num = jsonReqBody["batch_num"].asInt(); 	
		m_umParamT.regula = TypeDefs::RegulaType(jsonReqBody["regula"].asString().c_str()); 
		m_umParamT.learning_rate = jsonReqBody["learning_rate"].asDouble(); 
		m_rwLock.Unlock(); 

		pthread_t th_id;
		if(pthread_create(&th_id, NULL, Thread_OnceUpdateModel, NULL) == 0)
		{
			jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
			jsonResp["msg"] = "ok";
		}
		else
		{
			jsonResp["ret"] = _METIS_PLAT_UPDATESTART_FAIL; 
			jsonResp["msg"] = "start updating fail"; 
		}
	}
}


// 处理"loop_update_model"请求
void SlaveWorker::Work_LoopUpdateModel(Json::Value& jsonReqBody, Json::Value& jsonResp)
{
	bool is_updating; 

	m_rwLock.RdLock();
	is_updating = m_bIsUpdating; 
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING; 
		jsonResp["msg"] = "model updating";
	}
	else
	{
		m_rwLock.WrLock();
		m_umParamT.batch_num = jsonReqBody["batch_num"].asInt(); 	
		m_umParamT.regula = TypeDefs::RegulaType(jsonReqBody["regula"].asString().c_str()); 
		m_umParamT.learning_rate = jsonReqBody["learning_rate"].asDouble(); 
		m_rwLock.Unlock(); 

		pthread_t th_id;
		if(pthread_create(&th_id, NULL, Thread_LoopUpdateModel, NULL) == 0)
		{
			jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
			jsonResp["msg"] = "ok";
		}
		else
		{
			jsonResp["ret"] = _METIS_PLAT_UPDATESTART_FAIL; 
			jsonResp["msg"] = "start updating fail"; 
		}
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
		m_slaveTrainer.ReleaseModel(); 
		m_slaveTrainer.ReleasePatts(); 
	}
	m_rwLock.Unlock(); 

	if(is_updating)
	{
		jsonResp["ret"] = _METIS_PLAT_ISUPDATING;
		jsonResp["msg"] = "model updating";
	}
	else
	{
		jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
		jsonResp["msg"] = "ok"; 
		LOG_INFO("Slave trainer is released"); 
	}
}


// 处理"reset"请求
void SlaveWorker::Work_Reset(Json::Value& jsonResp)
{
	if(m_bIsUpdating)
	{
		m_slaveTrainer.UpdateCancel(); 
		m_slaveTrainer.ReleaseModel(); 
		m_slaveTrainer.ReleasePatts(); 
		m_bIsUpdating = false; 
	}
	jsonResp["ret"] = _METIS_PLAT_SUCCESS; 
	jsonResp["msg"] = "ok"; 
	LOG_INFO("Slave trainer is reset"); 
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


void* SlaveWorker::Thread_OnceUpdateModel(void* pParam)
{
	m_rwLock.WrLock();
	m_bIsUpdating = true;
	m_rwLock.Unlock();
	
	m_slaveTrainer.ModelOnceUpdate(m_umParamT.batch_num, m_umParamT.learning_rate, m_umParamT.regula); 
	
	m_rwLock.WrLock();
	m_bIsUpdating = false;
	m_rwLock.Unlock();
	return NULL; 
}


void* SlaveWorker::Thread_LoopUpdateModel(void* pParam)
{
	m_rwLock.WrLock();
	m_bIsUpdating = true;
	m_rwLock.Unlock();
	
	m_slaveTrainer.ModelLoopUpdate(m_umParamT.batch_num, m_umParamT.learning_rate, m_umParamT.regula); 
	
	m_rwLock.WrLock();
	m_bIsUpdating = false;
	m_rwLock.Unlock();
	return NULL; 
}


