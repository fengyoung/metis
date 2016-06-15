#ifndef _METIS_PLATFORM_MASTER_TRAINER_H 
#define _METIS_PLATFORM_MASTER_TRAINER_H 

#include <string>
#include <vector>
using namespace std; 
#include "TypeDefs.h"
using namespace metis_nn; 
#include "ThreadRWLock.h"
#include <stdint.h>

#define _UPDATING_WAIT_MICRO_SEC	(1000*100)


namespace metis_plat
{

typedef struct _thread_pattmapping_paramT
{
	string slave_hostport; 
	int32_t slave_id; 
	int32_t slave_cnt; 
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

	// mapping训练样本
	int32_t PattsMapping(vector<string>& vtrPattFiles);

	// 初始化模型
	virtual bool InitByLearningConfig(const char* sLearningConfFile) = 0;
	virtual bool InitByModel(const char* sModelFile) = 0; 

	// 执行训练
	virtual void Train(const char* sOutModelFile) = 0; 

protected:

	// 与所有slave同步模型
	bool SyncModel(); 	

	// 更新所有slave上的模型
	bool UpdateModel(const int32_t nBatchNum, const double dLearningRate, const ERegula eRegula); 

	// 从所有slave上收集更新后的模型数据
	virtual bool CollectModel() = 0;  	


protected: 

	// 检测一个slave是否可用
	bool DetectSlave(const char* sSlaveHostPort); 
	// 检测一个slave是否处于updating状态
	bool CheckSlaveUpdating(const char* sSlaveHostPort); 
	// 释放一个slave上的资源
	bool ReleaseSlave(const char* sSlaveHostPort); 
	// 设置一个slave上的模型数据	
	bool SetModelToSlave(const char* sSlaveHostPort, const char* sModelStr);
	// 更新一个slave上的模型
	bool UpdateModelInSlave(const char* sSlaveHostPort, const int32_t nBatchNum, const double dLearningRate, const ERegula eRegula); 
	// 获取一个slave上的模型数据
	bool GetModelFromSlave(const char* sSlaveHostPort, string& strModel, double& dAvgError);

	// 以字符串格式获取模型
	virtual string GetModelAsString() = 0; 

	// 按照输入的文件路径+文件名, 生成临时文件路径+文件名
	string TempFile(const char* sOutFile);

	// pattern mapping 线程
	static void* Thread_PattMapping(void* pParam);

protected: 
	vector<string> m_vtrSlaves;		// 可用的slave列表 
	double m_dAvgError; 

	static ThreadRWLock m_rwLock; 
	static int32_t m_nInput; 
	static int32_t m_nOutput; 
	static int32_t m_nPattCnt; 
};

}

#endif /* _METIS_PLATFORM_MASTER_TRAINER_H */

