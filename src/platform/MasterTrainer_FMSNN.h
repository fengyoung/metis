#ifndef _METIS_PLATFORM_MASTER_TRAINER_FMSNN_H 
#define _METIS_PLATFORM_MASTER_TRAINER_FMSNN_H 

#include "FMSNN.h"
using namespace metis_nn; 
#include "MasterTrainer.h"


namespace metis_plat
{

class MasterTrainer_FMSNN : public MasterTrainer
{
public:
	MasterTrainer_FMSNN(); 
	virtual ~MasterTrainer_FMSNN(); 

	// 初始化模型
	virtual bool InitByLearningConfig(const char* sLearningConfFile); 
	virtual bool InitByModel(const char* sModelFile); 

	// 执行训练
	virtual void Train(const char* sOutModelFile); 

protected:
	// 从所有slave上收集更新后的模型数据
	virtual bool CollectModel();

	// 以字符串格式获取模型
	virtual string GetModelAsString(); 

protected:
	FMSNN m_fmsnn;
};

}

#endif /* _METIS_PLATFORM_MASTER_TRAINER_FMSNN_H */

