#include "MasterTrainer.h"
#include "Worker.h"
using namespace metis_plat; 
#include <iostream>
#include <fstream>
using namespace std; 
#include "Pattern.h"
using namespace metis_nn; 
#include "StringArray.h"
#include "Random.h"
#include "TimeFmt.h"
#include "Timer.h"
using namespace metis_uti; 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>


ThreadRWLock MasterTrainer::m_rwLock; 
int32_t MasterTrainer::m_nInput = -1; 
int32_t MasterTrainer::m_nOutput = -1; 
int32_t MasterTrainer::m_nPattCnt = 0; 
ThPattMappingParamT MasterTrainer::m_thPattMappingParamT;


MasterTrainer::MasterTrainer()
{
	m_dTrainingLoss = 0.0; 
	m_dValidatedLoss = 0.0; 
}


MasterTrainer::~MasterTrainer()
{
	for(size_t i = 0; i < m_vtrValidatePatts.size(); i++) 
		delete m_vtrValidatePatts[i]; 
	m_vtrValidatePatts.clear(); 
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
	int32_t avaliable = 0, unavaliable = 0; 
	string str; 
	int32_t port, cnt; 
	char s_ip_port[64];  

	while(!ifs.eof())
	{
		std::getline(ifs, str); 
		if(str.empty())
			continue; 
		if(str.at(0) == '#')
			continue;

		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			continue;
		StringArray ar2(ar.GetString(1).c_str(), "_");
		if(ar2.Count() != 2)
			continue;  
		sscanf(ar2.GetString(0).c_str(), "%d", &port);
		sscanf(ar2.GetString(1).c_str(), "%d", &cnt);

		for(int32_t i = 0; i < cnt; i++) 
		{
			sprintf(s_ip_port, "%s:%d", ar.GetString(0).c_str(), port);
			if(DetectSlave(s_ip_port))
			{
				m_vtrSlaves.push_back(s_ip_port); 
				//printf("slave[%s] is avaliable.\n", str.c_str()); 
				avaliable++; 		
			}
			else
			{
				printf("slave[%s] is unavaliable.\n", s_ip_port); 
				unavaliable++; 		
			}
			port++; 	
		}	
	}
	ifs.close();
	printf("%d avaliable, %d unavaliable.\n", avaliable, unavaliable); 

	return !m_vtrSlaves.empty();
}


// 释放所有slave资源
void MasterTrainer::ReleaseSlaves()
{
	printf("** Release Slaves **\n"); 
	for(size_t i = 0; i < m_vtrSlaves.size(); i++) 
	{
		while(!ReleaseSlave(m_vtrSlaves[i].c_str()))
			usleep(_UPDATING_WAIT_MICRO_SEC); 
		//printf("slave[%s] has been released.\n", m_vtrSlaves[i].c_str()); 	
	}
}

// 释放所有slave上的模型
void MasterTrainer::ReleaseWeightInSlaves()
{
	printf("** Release Weight **\n"); 
	for(size_t i = 0; i < m_vtrSlaves.size(); i++) 
	{
		while(!ReleaseWeightInSlave(m_vtrSlaves[i].c_str()))
			usleep(_UPDATING_WAIT_MICRO_SEC); 
	}
}


// mapping训练样本
int32_t MasterTrainer::PattsMapping(vector<string>& vtrPattFiles, const int32_t nThreads)
{
	m_thPattMappingParamT.threads = nThreads; 
	m_thPattMappingParamT.slaves = m_vtrSlaves; 
	m_thPattMappingParamT.patt_files = vtrPattFiles; 
	Timer timer; 

	printf("** Patterns Mapping **\n"); 
	timer.Start();

	// 启动线程
	int32_t ids[nThreads];
	pthread_t th_id; 
	vector<pthread_t> vtr_thread_ids; 
	for(int32_t i = 0; i < nThreads; i++) 
	{
		ids[i] = i;  
		if(pthread_create(&th_id, NULL, Thread_PattMapping, &(ids[i])) == 0)
			vtr_thread_ids.push_back(th_id); 
	}
	// 等待线程结束
	for(size_t k = 0; k < vtr_thread_ids.size(); k++)
		pthread_join(vtr_thread_ids[k], NULL); 

	timer.Stop(); 
	printf("\rLoad %d patterns\n", m_nPattCnt); 
	printf("time_cost(s): %.3f\n", timer.GetLast_asSec());

	return m_nPattCnt; 
}


// 载入用于进行验证的样本
int32_t MasterTrainer::LoadValidatedPatts(const char* sValPattFile)
{
	for(size_t i = 0; i < m_vtrValidatePatts.size(); i++) 
		delete m_vtrValidatePatts[i]; 
	m_vtrValidatePatts.clear(); 

	ifstream ifs(sValPattFile); 
	if(!ifs.is_open())
		return -1; 
	string str; 
	Pattern* p_patt = NULL; 
	while(!ifs.eof())
	{
		std::getline(ifs, str); 
		if(str.empty())
			continue; 
		p_patt = new Pattern(); 
		if(!p_patt->FromString(str.c_str()))
		{
			delete p_patt; 
			continue; 
		}
		m_vtrValidatePatts.push_back(p_patt); 	
	}
	ifs.close();
 
	return (int32_t)m_vtrValidatePatts.size(); 
}


// 执行训练
bool MasterTrainer::Train(const char* sOutModelFile)
{
	NNAssi** pp_assi = CreateAssi(); 
	if(!pp_assi)
	{
		printf("Error, failed to create assistant parameters for training!\n"); 
		return false;  
	}
	LearnParams* p_learn_params = GetLearnParams(); 
	double learning_rate = p_learn_params->p_optim_params->learning_rate_init;	// learning rate 
	int32_t ss = 0; 
	double avg_loss, mini_loss = 99999999999.9;	// loss of one epoch 
	Timer timer;		// timer
	string str_tmp_file = TempFile(sOutModelFile); 
	bool tmp_flag = false; 

	int32_t early_stop = p_learn_params->early_stop; 
	if(m_vtrValidatePatts.empty()) 
		early_stop = 0; 
	else if(early_stop == 0) 
		early_stop = 10; 

	bool flag;	
	if(early_stop == 0)
	{ // early stop is disabled
		for(int32_t t = 0; t < p_learn_params->max_epoches; t++)
		{
			timer.Start(); 
#ifdef DEBUG 
			flag = Epoch(avg_loss, pp_assi, learning_rate, true); 
#else
			flag = Epoch(avg_loss, pp_assi, learning_rate, false); 
#endif 
			if(!flag)
			{
				ReleaseAssi(pp_assi); 
				printf("epoch %d | failed!!\n", t+1); 
				return false; 
			}
			ss++; 	
			if(avg_loss < mini_loss)
			{
				mini_loss = avg_loss; 
				ss = 0; 		
			}
			if(SaveTo(str_tmp_file.c_str()))
				tmp_flag = true; 
			else
			{
				tmp_flag = false; 
				printf("WARNING: failed to save TMP file!\n");
			}
			timer.Stop(); 	
			printf("epoch %d | training_loss: %.12g, ss: %d | time_cost(s): %.3f\n", 
					t+1, avg_loss, ss, timer.GetLast_asSec());
			if(ss >= 10 || avg_loss <= p_learn_params->epsilon)
				break; 	
		} 
	}
	else
	{ // early stop is enabled
		int32_t s = 0; 
		double validated_loss, auc, mini_validated_loss = 99999999999.9; 
	
		for(int32_t t = 0; t < p_learn_params->max_epoches; t++)
		{
			timer.Start(); 	
#ifdef DEBUG	
			flag = Epoch(avg_loss, pp_assi, learning_rate, true); 
#else
			flag = Epoch(avg_loss, pp_assi, learning_rate, false); 
#endif 
			if(!flag)
			{
				ReleaseAssi(pp_assi); 
				printf("epoch %d | failed!!\n", t+1); 
				return false; 
			}
			ss++; 	
			if(avg_loss < mini_loss)
			{
				mini_loss = avg_loss; 
				ss = 0; 		
			}
			timer.Stop(); 	
			
			// validation
			if(m_vtrValidatePatts[0]->m_nYCnt == 2)
			{
				Validation_Binary(validated_loss, auc); 
				s++; 	
				if(validated_loss < mini_validated_loss)
				{
					mini_validated_loss = validated_loss; 
					s = 0; 		
				}
				printf("epoch %d | training_loss: %.12g, ss: %d | validated_loss: %.12g, auc: %.6g, s: %d | time_cost(s): %.3f\n", 
						t+1, avg_loss, ss, validated_loss, auc, s, timer.GetLast_asSec()); 
			}
			else
			{
				validated_loss = Validation(); 
				s++; 	
				if(validated_loss < mini_validated_loss)
				{
					mini_validated_loss = validated_loss; 
					s = 0; 		
				}
				printf("epoch %d | training_loss: %.12g, ss: %d | validated_loss: %.12g, s: %d | time_cost(s): %.3f\n", 
						t+1, avg_loss, ss, validated_loss, s, timer.GetLast_asSec()); 
			}

			if(SaveTo(str_tmp_file.c_str()))
				tmp_flag = true; 
			else
			{
				tmp_flag = false; 
				printf("WARNING: failed to save TMP file!\n");
			}

			if(s >= early_stop || ss >= early_stop || avg_loss <= p_learn_params->epsilon)
				break; 	
		}
	}

	printf("** training done! time cost %.3f sec\n", timer.GetTotal_asSec());
	ReleaseAssi(pp_assi); 

	if(tmp_flag)
	{
		char scmd[256]; 
		sprintf(scmd, "mv -f %s %s", str_tmp_file.c_str(), sOutModelFile);
		system(scmd); 
	}
	else
	{
		if(!SaveTo(sOutModelFile))
		{
			printf("ERROR: failed to save model file!\n");
			return false; 
		} 
	}

	return true;  

}


// 查询训练样本信息
PattsInfoT MasterTrainer::QueryPattsInfo()
{
	PattsInfoT patts_info_t; 
	patts_info_t.x_dims = -1; 
	patts_info_t.y_dims = -1; 
	patts_info_t.patts = 0;
	Json::Value json_req_body, json_resp;

	for(size_t i = 0; i < m_vtrSlaves.size(); i++) 
	{
		if(Worker::SendCmd(m_vtrSlaves[i].c_str(), "patts_info", json_req_body, json_resp))
		{
			patts_info_t.x_dims = json_resp["x_dims"].asInt(); 
			patts_info_t.y_dims = json_resp["y_dims"].asInt(); 
			patts_info_t.patts += json_resp["patts"].asInt(); 
		}
	}
	m_nInput = patts_info_t.x_dims; 
	m_nOutput = patts_info_t.y_dims; 
	m_nPattCnt = patts_info_t.patts; 

	return patts_info_t; 
}


// 检测一个slave是否可用
bool MasterTrainer::DetectSlave(const char* sSlaveHostPort)
{
	Json::Value json_req_body, json_resp;
	return Worker::SendCmd(sSlaveHostPort, "detect", json_req_body, json_resp); 
}


// 检测一个slave是否处于updating状态
bool MasterTrainer::CheckSlaveUpdating(const char* sSlaveHostPort)
{
	Json::Value json_req_body, json_resp; 
	if(!Worker::SendCmd(sSlaveHostPort, "is_updating", json_req_body, json_resp))
		return false; 
	return json_resp["updating"].asBool(); 
}


// 释放一个slave上的资源
bool MasterTrainer::ReleaseSlave(const char* sSlaveHostPort)
{
	Json::Value json_req_body, json_resp; 
	if(!Worker::SendCmd(sSlaveHostPort, "release", json_req_body, json_resp))
		return true;
	return (json_resp["ret"].asInt() == _METIS_PLAT_SUCCESS);
}


// 释放一个slave上的权重矩阵
bool MasterTrainer::ReleaseWeightInSlave(const char* sSlaveHostPort)
{
	Json::Value json_req_body, json_resp; 
	if(!Worker::SendCmd(sSlaveHostPort, "release_weight", json_req_body, json_resp))
		return true;
	return (json_resp["ret"].asInt() == _METIS_PLAT_SUCCESS);
}


// 从一个slave上获取梯度计算结果
bool MasterTrainer::GetBatchGradFromSalve(const char* sSlaveHostPort, char* stream, int32_t& len, const int32_t nBufSize)
{
	while(true)
	{
		usleep(_UPDATING_WAIT_MICRO_SEC); 
		if(!Worker::GetBatchGrad(sSlaveHostPort, stream, len, nBufSize))
			return false; 
		if(strncmp(stream, "updating##", strlen("updating##")) != 0)
			break; 
	}
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


void* MasterTrainer::Thread_PattMapping(void* pParam)
{
	//pthread_detach(pthread_self());

	int32_t id = *((int32_t*)pParam); 
	int32_t slave_cnt = (int32_t)m_thPattMappingParamT.slaves.size(); 

	Json::Value json_req_body, json_resp; 
	string str_patt; 
	int32_t cnt = 0, success_cnt = 0; 
	int32_t slave_id; 
	bool first_flag = true; 

	for(int32_t i = 0; i < (int32_t)m_thPattMappingParamT.patt_files.size(); i++) 
	{
		ifstream ifs(m_thPattMappingParamT.patt_files[i].c_str()); 
		if(!ifs.is_open())
		{
			m_rwLock.WrLock();
			printf("[thread %d] failed to open %s\n", id, m_thPattMappingParamT.patt_files[i].c_str()); 
			m_rwLock.Unlock(); 
			continue; 
		}
		while(!ifs.eof())
		{
			std::getline(ifs, str_patt); 
			if(str_patt.empty())
				continue; 

			if(first_flag)
			{
				Pattern patt; 
				if(patt.FromString(str_patt.c_str()))
				{
					m_rwLock.WrLock();
					m_nInput = patt.m_nXCnt; 
					m_nOutput = patt.m_nYCnt; 
					m_rwLock.Unlock(); 
					first_flag = false;
				} 
			}

			if(cnt % m_thPattMappingParamT.threads == id)
			{
				m_rwLock.WrLock();
				slave_id = Random::RandUni_Int(0, slave_cnt); 
				m_rwLock.Unlock(); 
				if(Worker::SendBitStream(m_thPattMappingParamT.slaves[slave_id].c_str(), "push_patt_string", str_patt.c_str(), str_patt.length(), json_resp))
				{
					if(json_resp["ret_code"].asInt() == _METIS_PLAT_SUCCESS)
						success_cnt++; 
				}
			}
			cnt++; 

			if(cnt % 128 == 0)
			{
				m_rwLock.WrLock();
				m_nPattCnt += success_cnt; 
				printf("\rLoad %d patterns", m_nPattCnt); 
				m_rwLock.Unlock();
				success_cnt = 0; 	
			} 	
		}

		ifs.close(); 
	}

	m_rwLock.WrLock();
	m_nPattCnt += success_cnt; 
	m_rwLock.Unlock(); 

//	pthread_exit(0);
	return NULL; 
}


