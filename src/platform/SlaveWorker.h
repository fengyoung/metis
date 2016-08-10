#ifndef _METIS_PLATFORM_SLAVE_WORKER_H 
#define _METIS_PLATFORM_SLAVE_WORKER_H 

#include <string>
#include <vector>
using namespace std; 
#include "SlaveTrainer.h"
#include "Worker.h"


namespace metis_plat
{


typedef struct _update_model_param_t
{
	int32_t batch_num;  	
	ERegula regula; 
	double learning_rate; 
} UpdateModelParamT;


class SlaveWorker : public Worker 
{
public: 
	SlaveWorker(); 
	virtual ~SlaveWorker(); 

	void WorkCore(Json::Value& jsonReq, Json::Value& jsonResp); 

	void Work_RecvBinaryPatt(const char* bitStream, const int32_t nLen, Json::Value& jsonResp); 

private:
	// 处理"detect"请求
	//
	// REQ:   {"cmd":"detect", "body":{}}
	// RESP:  {"ret":0, "msg":"ok"}
	//
	void Work_Detect(Json::Value& jsonResp); 

	// 处理"push_patt"请求
	//
	// REQ:   {"cmd":"push_patt", "body":{"patt":"..."}}
	// RESP:  {"ret":0, "msg":"ok"}
	//        {"ret":1, "msg":"model updating"}
	//        {"ret":-101, "msg":"push fail"}
	//
	void Work_PushPatt(Json::Value& jsonReqBody, Json::Value& jsonResp); 
	
	// 处理"get_patt_cnt"请求
	//
	// REQ:   {"cmd":"get_patt_cnt", "body":{}}
	// RESP:  {"ret":0, "msg":"ok", "patts":19029}
	//
	void Work_GetPattCnt(Json::Value& jsonResp); 

	// 处理"shuffle_patt"请求
	//
	// REQ:   {"cmd":"shuffle_patt", "body":{}}
	// RESP:  {"ret":0, "msg":"ok"}
	//        {"ret":1, "msg":"model updating"}
	//
	void Work_ShufflePatt(Json::Value& jsonResp); 

	// 处理"set_model"请求
	//
	// REQ:   {"cmd":"set_model", "body":{"model":"..."}}
	// RESP:  {"ret":0, "msg":"ok"}
	//        {"ret":1, "msg":"model updating"}
	//        {"ret":-102, "msg":"set fail"}
	//
	void Work_SetModel(Json::Value& jsonReqBody, Json::Value& jsonResp); 
	
	// 处理"get_model"请求
	//
	// REQ:   {"cmd":"get_model", "body":{}}
	// RESP:  {"ret":0, "msg":"ok", "avg_error":0.027618, "model":"..."}
	//        {"ret":1, "msg":"model updating"}
	//        {"ret":-103, "msg":"get fail"}
	//
	void Work_GetModel(Json::Value& jsonResp); 

	// 处理"once_update_model"请求
	//
	// REQ:   {"cmd":"once_update_model", "body":{"batch_num":500, "learning_rate":0.05, "regula":"L1"}}
	// RESP:  {"ret":0, "msg":"ok"}
	//        {"ret":1, "msg":"model updating"}
	//        {"ret":-104, "msg":"start updating fail"}
	//
	void Work_OnceUpdateModel(Json::Value& jsonReqBody, Json::Value& jsonResp); 
	
	// 处理"loop_update_model"请求
	//
	// REQ:   {"cmd":"loop_update_model", "body":{"batch_num":500, "learning_rate":0.05, "regula":"L1"}}
	// RESP:  {"ret":0, "msg":"ok"}
	//        {"ret":1, "msg":"model updating"}
	//        {"ret":-104, "msg":"start updating fail"}
	//
	void Work_LoopUpdateModel(Json::Value& jsonReqBody, Json::Value& jsonResp); 

	// 处理"release"请求
	//
	// REQ:   {"cmd":"release", "body":{}}
	// RESP:  {"ret":0, "msg":"ok"}
	//        {"ret":1, "msg":"model updating"}
	//
	void Work_Release(Json::Value& jsonResp); 
	
	// 处理"reset"请求
	//
	// REQ:   {"cmd":"reset", "body":{}}
	// RESP:  {"ret":0, "msg":"ok"}
	//
	void Work_Reset(Json::Value& jsonResp); 

	// 处理"is_updating"请求
	//
	// REQ:   {"cmd":"is_updating", "body":{}}
	// RESP:  {"ret":0, "msg":"ok", "updating":true}
	//
	void Work_IsUpdating(Json::Value& jsonResp); 
	
	static void* Thread_OnceUpdateModel(void* pParam); 
	static void* Thread_LoopUpdateModel(void* pParam); 

private: 
	static SlaveTrainer m_slaveTrainer; 
	static bool m_bIsUpdating;
	static UpdateModelParamT m_umParamT; 
};


}

#endif /* _METIS_PLATFORM_SLAVE_WORKER_H */

