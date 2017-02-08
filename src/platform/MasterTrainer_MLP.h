#ifndef _METIS_PLATFORM_MASTER_TRAINER_MLP_H 
#define _METIS_PLATFORM_MASTER_TRAINER_MLP_H 

#include "MLP.h"
using namespace metis_nn; 
#include "MasterTrainer.h"


namespace metis_plat
{

class MasterTrainer_MLP : public MasterTrainer
{
public:
	MasterTrainer_MLP(); 
	virtual ~MasterTrainer_MLP(); 

	// 初始化模型
	virtual bool InitByLearningConfig(const char* sLearningConfFile); 
	virtual bool InitByModel(const char* sModelFile); 

	// 一轮训练
	virtual bool Epoch(double& dAvgLoss, NNAssi** ppAssi, const double dLearningRate, const bool bValid = false);

	// 一次迭代
	virtual bool Iteration(double& dLoss, NNAssi** ppAssi, const double dLearningRate); 

	// 保存模型
	virtual bool SaveTo(const char* sModelFile); 	

	virtual void PrintParams(ostream& os);

protected:
	// 同步权重向量
	virtual bool SyncWeight_toAllSlaves(); 
	// 更新计算所有slave上的梯度
	virtual bool CalcBatchGrad_inAllSlaves(); 
	// 收集所有slave上的梯度计算结果
	virtual bool CollectBatchGrad_fromAllSlaves(NNAssi** ppAssi, double& dAvgLoss); 
 
	virtual NNAssi** CreateAssi(); 
	virtual void ReleaseAssi(NNAssi** ppAssi); 
	virtual LearnParams* GetLearnParams(); 
	virtual double Validation(); 
	virtual int32_t Validation_Binary(double& dValidatedLoss, double& dAuc); 
	
protected:
	MLP m_mlp; 
};

}

#endif /* _METIS_PLATFORM_MASTER_TRAINER_MLP_H */

