#ifndef _METIS_PLATFORM_MASTER_TRAINER_H 
#define _METIS_PLATFORM_MASTER_TRAINER_H 

#include <string>
#include <vector>
#include <iostream>
using namespace std; 
#include "NNAssi.h"
using namespace metis_nn; 
#include "ThreadRWLock.h"
#include "SlaveTrainer.h"
#include <stdint.h>

#define _UPDATING_WAIT_MICRO_SEC	(500)
#define _DEFAULT_MAX_THREADS		32

namespace metis_plat
{

typedef struct _thread_pattmapping_paramT
{
	int32_t threads; 
	vector<string> slaves; 
	vector<string> patt_files; 
} ThPattMappingParamT; 


class MasterTrainer
{
public:
	MasterTrainer(); 
	virtual ~MasterTrainer(); 

	// 载入slave列表, 并检测
	bool LoadSlaves(const char* sSlaveListFile); 

	// 释放所有slave资源
	void ReleaseSlaves(); 
	
	// 释放所有slave上的权重
	void ReleaseWeightInSlaves();

	// mapping训练样本
	int32_t PattsMapping(vector<string>& vtrPattFiles, const int32_t nThreads = _DEFAULT_MAX_THREADS);

	// 载入用于进行验证的样本
	int32_t LoadValidatedPatts(const char* sValPattFile);
	
	// 初始化模型
	virtual bool InitByLearningConfig(const char* sLearningConfFile) = 0;
	virtual bool InitByModel(const char* sModelFile) = 0; 

	// 执行训练
	virtual bool Train(const char* sOutModelFile); 
	
	// 一轮训练
	virtual bool Epoch(double& dAvgLoss, NNAssi** ppAssi, const double dLearningRate, const bool bValid = false) = 0;
	// 一次迭代
	virtual bool Iteration(double& dLoss, NNAssi** ppAssi, const double dLearningRate) = 0; 
	
	// 保存模型
	virtual bool SaveTo(const char* sModelFile) = 0; 	

	virtual void PrintParams(ostream& os) = 0; 
	
	// 查询训练样本信息
	PattsInfoT QueryPattsInfo(); 

protected:
	// 同步权重向量
	virtual bool SyncWeight_toAllSlaves() = 0; 
	// 更新计算所有slave上的梯度
	virtual bool CalcBatchGrad_inAllSlaves() = 0;
	// 收集所有slave上的梯度计算结果
	virtual bool CollectBatchGrad_fromAllSlaves(NNAssi** ppAssi, double& dAvgLoss) = 0;

	virtual NNAssi** CreateAssi() = 0; 
	virtual void ReleaseAssi(NNAssi** ppAssi) = 0; 
	virtual LearnParams* GetLearnParams() = 0; 
	virtual double Validation() = 0; 
	virtual int32_t Validation_Binary(double& dValidatedLoss, double& dAuc) = 0; 

protected: 
	// 检测一个slave是否可用
	bool DetectSlave(const char* sSlaveHostPort); 
	// 检测一个slave是否处于updating状态
	bool CheckSlaveUpdating(const char* sSlaveHostPort); 
	// 释放一个slave上的资源
	bool ReleaseSlave(const char* sSlaveHostPort); 
	// 释放一个slave上的权重矩阵
	bool ReleaseWeightInSlave(const char* sSlaveHostPort); 
	// 从一个slave上获取梯度计算结果
	bool GetBatchGradFromSalve(const char* sSlaveHostPort, char* stream, int32_t& len, const int32_t nBufSize);
	
	// 按照输入的文件路径+文件名, 生成临时文件路径+文件名
	string TempFile(const char* sOutFile);

	// pattern mapping 线程
	static void* Thread_PattMapping(void* pParam);

protected: 
	vector<string> m_vtrSlaves;		// 可用的slave列表 
	double m_dTrainingLoss; 
	double m_dValidatedLoss; 

	vector<Pattern*> m_vtrValidatePatts;	// 用于进行验证的样本

	static ThreadRWLock m_rwLock; 
	static int32_t m_nInput; 
	static int32_t m_nOutput; 
	static int32_t m_nPattCnt; 

	static ThPattMappingParamT m_thPattMappingParamT; 
};

}

#endif /* _METIS_PLATFORM_MASTER_TRAINER_H */

