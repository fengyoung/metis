#ifndef _METIS_PLATFORM_SLAVE_WORKER_H 
#define _METIS_PLATFORM_SLAVE_WORKER_H 

#include <string>
#include <vector>
using namespace std; 
#include "SlaveTrainer.h"
#include "Worker.h"


namespace metis_plat
{


typedef struct _calc_grad_params_t
{
	int32_t batch_size;  
	EActType hidden_act; 
} CalcGradParamsT;



class SlaveWorker : public Worker 
{
public: 
	SlaveWorker(); 
	virtual ~SlaveWorker(); 

	void WorkCore(Json::Value& jsonReq, Json::Value& jsonResp); 
	void WorkCore_bitStream(const char* bitStream, const int32_t nLen, Json::Value& jsonResp); 
	void WorkCore_GetBatchGrad(char* bitStream_Grad, int32_t& nLen); 

private:
	// 处理"detect"请求
	//
	// REQ:   {"cmd":"detect", "body":{}}
	// RESP:  {"ret":0, "msg":"ok"}
	//
	void Work_Detect(Json::Value& jsonResp); 
	
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

	// 处理"calc_grad"请求
	//
	// REQ:   {"cmd":"calc_grad", "body":{"batch_size":200, "hidden_act":"relu"}}
	// RESP:  {"ret":0, "msg":"ok"}
	//        {"ret":1, "msg":"model updating"}
	//        {"ret":-104, "msg":"start calculation fail"}
	//
	void Work_CalcGrad(Json::Value& jsonReqBody, Json::Value& jsonResp); 

	// 处理"release_weight"请求
	//
	// REQ:   {"cmd":"release_model", "body":{}}
	// RESP:  {"ret":0, "msg":"ok"}
	//        {"ret":1, "msg":"model updating"}
	//
	void Work_ReleaseWeight(Json::Value& jsonResp);
	
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
	
	// 处理"patts_info"请求
	//
	// REQ:   {"cmd":"patts_info", "body":{}}
	// RESP:  {"ret":0, "msg":"ok", "x_dims":3092, "y_dims":2, "patts":3982712}
	//
	void Work_PattsInfo(Json::Value& jsonResp); 

	// 处理"push_patt"请求
	void Work_PushPatt(const char* bitStream_Patt, const int32_t nLen, Json::Value& jsonResp); 
	// 处理"push_patt_string"请求
	void Work_PushPattString(const char* sPattStr, Json::Value& jsonResp); 
	// 处理"set_weight"请求
	void Work_SetWeight(const char* bitStream_Weight, const int32_t nLen, Json::Value& jsonResp); 

	
	static void* Thread_CalcBatchGrad(void* pParam); 

private: 
	static SlaveTrainer m_slaveTrainer; 
	static bool m_bIsUpdating;
	static double m_dAvgLoss; 
	static CalcGradParamsT m_calcGradParamsT; 
};


}

#endif /* _METIS_PLATFORM_SLAVE_WORKER_H */

