#include "MasterTrainer_MLP.h"
#include "JsonTransf.h"
#include "Worker.h"
using namespace metis_plat; 
#include "Timer.h"
using namespace metis_uti; 
#include <iostream>
using namespace std; 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>


MasterTrainer_MLP::MasterTrainer_MLP() : MasterTrainer()
{
}


MasterTrainer_MLP::~MasterTrainer_MLP()
{
}


bool MasterTrainer_MLP::InitByLearningConfig(const char* sLearningConfFile)
{
	if(m_nInput < 0 || m_nOutput < 0)
		return false; 
	return m_mlp.InitFromConfig(sLearningConfFile, m_nInput, m_nOutput); 
}


bool MasterTrainer_MLP::InitByModel(const char* sModelFile)
{
	if(m_mlp.Load(sModelFile) != _METIS_NN_SUCCESS)
		return false;
	if(m_nInput > 0 && m_nOutput > 0)
	{
		if(m_mlp.GetArchParams()->input != m_nInput || m_mlp.GetArchParams()->output != m_nOutput)
			return false; 
	}
	return true; 
}


// 一轮训练
bool MasterTrainer_MLP::Epoch(double& dAvgLoss, NNAssi** ppAssi, const double dLearningRate, const bool bValid) 
{
	if(m_mlp.IsNull() || !ppAssi)
		return false; 
	int32_t iter = 0; 
	LearnParams* p_learn_params = m_mlp.GetLearnParams(); 	
	double avg_loss, validated_loss, auc; 
	dAvgLoss = 0.0; 
	
	while(iter * (int32_t)m_vtrSlaves.size() * p_learn_params->batch_size < m_nPattCnt)
	{
		Timer timer; 
		timer.Start(); 
		if(!Iteration(avg_loss, ppAssi, dLearningRate))
			return false; 
		dAvgLoss += avg_loss; 
		timer.Stop(); 	
		
		if(bValid)
		{	
			if(m_vtrValidatePatts[0]->m_nYCnt == 2)
			{
				Validation_Binary(validated_loss, auc);
				printf("\titer %d | training_loss: %.12g | validated_loss: %.12g, auc: %.6g | time_cost(ms): %.3f\n", 
						iter+1, avg_loss, validated_loss, auc, timer.GetTotal_asMSec()); 
			}
			else
			{
				validated_loss = Validation(); 
				printf("\titer %d | training_loss: %.12g | validated_loss: %.12g | time_cost(ms): %.3f\n", 
						iter+1, avg_loss, validated_loss, timer.GetTotal_asMSec()); 
			} 
		}

		iter++; 
	}

	dAvgLoss /= (double)iter; 
	return true; 
} 


// 一次迭代
bool MasterTrainer_MLP::Iteration(double& dLoss, NNAssi** ppAssi, const double dLearningRate)
{
	if(m_mlp.IsNull() || !ppAssi)
		return false; 

	// 同步权重向量
	if(!SyncWeight_toAllSlaves())
		return false;
	
	// 更新计算所有slave上的梯度
	if(!CalcBatchGrad_inAllSlaves())
		return false; 
	
	// 收集所有slave上的梯度计算结果
	if(!CollectBatchGrad_fromAllSlaves(ppAssi, dLoss))
		return false; 
	
	// 更新权重
	if(!m_mlp.ModelUpdate(ppAssi, dLearningRate))
		return false; 
	
	return true; 
}


// 保存模型
bool MasterTrainer_MLP::SaveTo(const char* sModelFile)
{
	if(!sModelFile)
		return false; 
	return (m_mlp.Save(sModelFile) == _METIS_NN_SUCCESS); 
}


void MasterTrainer_MLP::PrintParams(ostream& os)
{
	m_mlp.PrintParams(os); 
}


// 同步权重向量
bool MasterTrainer_MLP::SyncWeight_toAllSlaves()
{
	if(m_mlp.IsNull())
		return false; 
	int32_t hl = (int32_t)((ArchParams_MLP*)m_mlp.GetArchParams())->vtr_hiddens.size(); 	
	int32_t len = Matrix::MatsStreamSize(m_mlp.m_ws, hl+1); 
	char* stream = new char[len + 1]; 
	len = Matrix::MatsToStream(stream, m_mlp.m_ws, hl+1);
	Json::Value json_resp;  
	int32_t ret; 
	
	for(size_t i = 0; i < m_vtrSlaves.size(); i++) 
	{
		while(true)
		{
			if(!Worker::SendBitStream(m_vtrSlaves[i].c_str(), "set_weight", stream, len, json_resp))
			{
				delete stream; 
				return false; 
			}
			ret = json_resp["ret"].asInt(); 
			if(ret == _METIS_PLAT_SUCCESS)
				break; 
			else if(ret == _METIS_PLAT_ISUPDATING)
				usleep(_UPDATING_WAIT_MICRO_SEC); 
			else
			{
				delete stream; 
				return false; 
			}
		}
	}
	delete stream; 
	return true; 
}


// 更新计算所有slave上的梯度
bool MasterTrainer_MLP::CalcBatchGrad_inAllSlaves()
{
	if(m_mlp.IsNull())
		return false; 

	LearnParams* p_learn_params = m_mlp.GetLearnParams(); 	
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_mlp.GetArchParams(); 	

	Json::Value json_req_body, json_resp;
	json_req_body["batch_size"] = p_learn_params->batch_size; 
	json_req_body["hidden_act"] = ActConv::ActName(p_arch_params->hidden_act); 
	int32_t ret; 

	for(size_t i = 0; i < m_vtrSlaves.size(); i++)
	{
		while(true)
		{
			if(!Worker::SendCmd(m_vtrSlaves[i].c_str(), "calc_grad", json_req_body, json_resp))
				return false; 	
			ret = json_resp["ret"].asInt(); 
			if(ret == _METIS_PLAT_SUCCESS)
				break; 
			else if(ret == _METIS_PLAT_ISUPDATING)
				usleep(_UPDATING_WAIT_MICRO_SEC); 
			else
				return false;
		} 
	}

	return true; 
}


// 收集所有slave上的梯度计算结果
bool MasterTrainer_MLP::CollectBatchGrad_fromAllSlaves(NNAssi** ppAssi, double& dAvgLoss)
{
	if(m_mlp.IsNull() || !ppAssi)
		return false;

	int32_t hl = (int32_t)((ArchParams_MLP*)m_mlp.GetArchParams())->vtr_hiddens.size(); 	
	int32_t len, buf_size = Matrix::MatsStreamSize(m_mlp.m_ws, hl+1) + 100; 
	char* stream = new char[buf_size]; 
	double loss;
	int32_t off, mat_cnt, cnt = 0; 
	Matrix* gs = NULL; 
	dAvgLoss = 0.0; 
	for(size_t i = 0; i < m_vtrSlaves.size(); i++)
	{
		off = 0;  
		if(GetBatchGradFromSalve(m_vtrSlaves[i].c_str(), stream, len, buf_size))
		{
			loss = *((double*)(stream + off));
			off += sizeof(double);
			gs = Matrix::MatsFromStream(mat_cnt, stream + off, len - off); 
			if(gs)
			{
				if(cnt == 0)
				{
					for(int32_t h = 0; h <= hl; h++) 
					{
						ppAssi[h]->m_g.CombineWith(gs[h], 0.0, 1.0);
					}
				}
				else
				{
					for(int32_t h = 0; h <= hl; h++) 
						ppAssi[h]->m_g.CombineWith(gs[h], 1.0, 1.0);
				}
				cnt++; 
				dAvgLoss += loss; 
				delete [] gs; 
			}
		}
	}	
	dAvgLoss /= (double)cnt; 

	return true; 
}


NNAssi** MasterTrainer_MLP::CreateAssi()
{
	return m_mlp.CreateAssi(true); 
}


void MasterTrainer_MLP::ReleaseAssi(NNAssi** ppAssi)
{
	if(ppAssi)
	{
		int32_t hl = (int32_t)((ArchParams_MLP*)m_mlp.GetArchParams())->vtr_hiddens.size(); 	
		for(int32_t h = 0; h <= hl; h++)	
			delete ppAssi[h]; 
		delete [] ppAssi; 
	}
}


LearnParams* MasterTrainer_MLP::GetLearnParams()
{
	return m_mlp.GetLearnParams(); 
}


double MasterTrainer_MLP::Validation()
{
	return m_mlp.Validation(m_vtrValidatePatts, (int32_t)m_vtrValidatePatts.size()); 
}


int32_t MasterTrainer_MLP::Validation_Binary(double& dValidatedLoss, double& dAuc)
{
	return m_mlp.Validation_Binary(dValidatedLoss, dAuc, m_vtrValidatePatts, (int32_t)m_vtrValidatePatts.size()); 
}



