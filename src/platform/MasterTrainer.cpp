#include "MasterTrainer.h"
#include "Worker.h"
using namespace metis_plat; 
#include <iostream>
#include <fstream>
using namespace std; 
#include "Pattern.h"
using namespace metis_nn; 
#include "StringArray.h"
#include "TimeFmt.h"
#include "Timer.h"
using namespace metis_uti; 
#include <stdio.h>


ThreadRWLock MasterTrainer::m_rwLock; 
int32_t MasterTrainer::m_nInput = -1; 
int32_t MasterTrainer::m_nOutput = -1; 
int32_t MasterTrainer::m_nPattCnt = 0; 



MasterTrainer::MasterTrainer()
{
}


MasterTrainer::~MasterTrainer()
{
}


// 载入slave列表, 并检测
bool MasterTrainer::LoadSlaves(const char* sSlaveListFile)
{
	printf("** Detect Slaves **\n"); 
	ifstream ifs(sSlaveListFile); 
	if(!ifs.is_open())
	{
		printf("failed to open slave list file %s\n", sSlaveListFile); 
		return false; 
	}
	m_vtrSlaves.clear(); 
	string str; 
	while(!ifs.eof())
	{
		std::getline(ifs, str); 
		if(str.empty())
			continue; 
		if(str.at(0) == '#')
			continue; 
		if(DetectSlave(str.c_str()))
			m_vtrSlaves.push_back(str); 
	}
	ifs.close();

	if(m_vtrSlaves.empty())
	{
		printf("no avaliable slave.\n"); 
		return false; 
	}
	for(size_t i = 0; i < m_vtrSlaves.size(); i++) 
	{
		printf("slave[%s] is avaliable.\n", m_vtrSlaves[i].c_str());
	}

	return true; 
}


// 释放所有slave资源
void MasterTrainer::ReleaseSlaves()
{
	printf("** Release Slaves **\n"); 
	for(size_t i = 0; i < m_vtrSlaves.size(); i++) 
	{
		while(!ReleaseSlave(m_vtrSlaves[i].c_str()))
			usleep(_UPDATING_WAIT_MICRO_SEC); 
		printf("slave[%s] has been released.\n", m_vtrSlaves[i].c_str()); 	
	}
}


// mapping训练样本
int32_t MasterTrainer::PattsMapping(vector<string>& vtrPattFiles)
{
	pthread_t th_id;
	vector<pthread_t> vtr_thread_ids;
	ThPattMappingParamT th_param;  
	vector<ThPattMappingParamT> vtr_param_s;	
	Timer timer; 

	// 构造参数
	th_param.slave_id = -1; 
	th_param.slave_cnt = (int32_t)m_vtrSlaves.size(); 
	th_param.patt_files = vtrPattFiles; 
	for(size_t i = 0; i < m_vtrSlaves.size(); i++)  
	{
		th_param.slave_hostport = m_vtrSlaves[i]; 
		th_param.slave_id += 1; 
		vtr_param_s.push_back(th_param); 
	}

	printf("** Patterns Mapping **\n"); 
	timer.Start();

	// 启动线程
	for(size_t j = 0; j < vtr_param_s.size(); j++) 
	{
		if(pthread_create(&th_id, NULL, Thread_PattMapping, &(vtr_param_s[j])) == 0)
			vtr_thread_ids.push_back(th_id); 
	}

	// 等待线程结束
	for(size_t k = 0; k < vtr_thread_ids.size(); k++)
		pthread_join(vtr_thread_ids[k], NULL); 

	timer.Stop();
	printf("time_cost(s): %.3f\n", timer.GetLast_asSec());


	return m_nPattCnt; 
}


// 与所有slave同步模型
bool MasterTrainer::SyncModel()
{
	string str_model = GetModelAsString(); 
	for(size_t i = 0; i < m_vtrSlaves.size(); i++) 
	{
		if(!SetModelToSlave(m_vtrSlaves[i].c_str(), str_model.c_str()))
			return false; 
	}
	return true; 
}


// 更新所有slave上的模型
bool MasterTrainer::UpdateModel(const int32_t nBatchNum, const double dLearningRate, const ERegula eRegula)
{
	for(size_t i = 0; i < m_vtrSlaves.size(); i++) 
	{
		if(!UpdateModelInSlave(m_vtrSlaves[i].c_str(), nBatchNum, dLearningRate, eRegula))
			return false; 
	}
	return true; 
}





// 检测一个slave是否可用
bool MasterTrainer::DetectSlave(const char* sSlaveHostPort) 
{
	Json::Value json_req_body, json_resp; 
	return Worker::Send(sSlaveHostPort, "detect", json_req_body, json_resp); 
}


// 检测一个slave是否处于updating状态
bool MasterTrainer::CheckSlaveUpdating(const char* sSlaveHostPort)
{
	Json::Value json_req_body, json_resp; 
	if(!Worker::Send(sSlaveHostPort, "is_updating", json_req_body, json_resp))
		return false; 
	return json_resp["updating"].asBool(); 
}


// 释放一个slave上的资源
bool MasterTrainer::ReleaseSlave(const char* sSlaveHostPort)
{
	Json::Value json_req_body, json_resp; 
	if(!Worker::Send(sSlaveHostPort, "release", json_req_body, json_resp))
		return true; 
	return (json_resp["ret"].asInt() == _METIS_PLAT_SUCCESS);
}


// 设置一个slave上的模型数据	
bool MasterTrainer::SetModelToSlave(const char* sSlaveHostPort, const char* sModelStr)
{
	Json::Value json_req_body, json_resp; 
	json_req_body["model"] = sModelStr; 
	if(!Worker::Send(sSlaveHostPort, "set_model", json_req_body, json_resp))
		return false; 
	return (json_resp["ret"].asInt() == _METIS_PLAT_SUCCESS);
}


// 更新一个slave上的模型
bool MasterTrainer::UpdateModelInSlave(const char* sSlaveHostPort, const int32_t nBatchNum, const double dLearningRate, const ERegula eRegula)
{
	Json::Value json_req_body, json_resp; 
	json_req_body["batch_num"] = nBatchNum; 
	json_req_body["learning_rate"] = dLearningRate; 
	json_req_body["regula"] = TypeDefs::RegulaName(eRegula); 
	if(!Worker::Send(sSlaveHostPort, "loop_update_model", json_req_body, json_resp))
		return false; 
	return (json_resp["ret"].asInt() == _METIS_PLAT_SUCCESS);
}


// 获取一个slave上的模型数据
bool MasterTrainer::GetModelFromSlave(const char* sSlaveHostPort, string& strModel, double& dAvgError)
{
	Json::Value json_req_body, json_resp; 
	if(!Worker::Send(sSlaveHostPort, "get_model", json_req_body, json_resp))
		return false; 
	while(json_resp["ret"].asInt() == _METIS_PLAT_ISUPDATING)
	{
		usleep(_UPDATING_WAIT_MICRO_SEC); 	
		if(!Worker::Send(sSlaveHostPort, "get_model", json_req_body, json_resp))
			return false; 
	}
	if(json_resp["ret"].asInt() != _METIS_PLAT_SUCCESS)
		return false; 
	strModel = json_resp["model"].asString();
	dAvgError = json_resp["avg_error"].asDouble();
	return true; 
}


// 按照输入的文件路径+文件名, 生成临时文件路径+文件名
string MasterTrainer::TempFile(const char* sOutFile)
{
	string tmp_file(sOutFile);
	tmp_file += ".METIS.TMP.";
	tmp_file += TimeFmt::CurTime_asStr(_TIME_FMT_COMPACT);
	return tmp_file;
}


// pattern mapping 线程
void* MasterTrainer::Thread_PattMapping(void* pParam)
{
	ThPattMappingParamT* pparam = (ThPattMappingParamT*)pParam; 
	Json::Value json_req_body, json_resp; 
	string str_patt; 
	int32_t lines, cnt = 0; 
	bool first_flag = true; 

	for(int32_t i = 0; i < (int32_t)pparam->patt_files.size(); i++) 
	{
		ifstream ifs(pparam->patt_files[i].c_str()); 
		if(!ifs.is_open())
		{
			m_rwLock.WrLock();
			printf("[thread %d] failed to open %s\n", pparam->slave_id, pparam->patt_files[i].c_str()); 	
			m_rwLock.Unlock(); 
			continue; 
		}
		
		lines = 0; 
		while(!ifs.eof())
		{
			std::getline(ifs, str_patt); 
			if(str_patt.empty())
				continue; 
			if(first_flag) 
			{
				m_rwLock.WrLock();
				if(m_nInput < 0 && m_nOutput < 0)
				{
					Pattern patt;
					if(patt.FromString(str_patt.c_str()))
					{
						m_nInput = patt.m_nXCnt; 	
						m_nOutput = patt.m_nYCnt; 	
						first_flag = false; 
					}
				}
				m_rwLock.Unlock(); 
			}
			if(lines % pparam->slave_cnt == pparam->slave_id)
			{
				json_req_body.clear(); 
				json_req_body["patt"] = str_patt; 
				if(Worker::Send(pparam->slave_hostport.c_str(), "push_patt", json_req_body, json_resp))
				{
					if(json_resp["ret_code"].asInt() == _METIS_PLAT_SUCCESS)
						cnt++; 
				}
			}
			lines++; 
		}

		ifs.close(); 
	}
	m_rwLock.WrLock();
	m_nPattCnt += cnt; 
	printf("[thread %d] %d patterns have been send to slave[%s] \n", pparam->slave_id, cnt, pparam->slave_hostport.c_str()); 
	m_rwLock.Unlock(); 

	return NULL; 


}




