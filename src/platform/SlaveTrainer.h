#ifndef _METIS_PLATFORM_SLAVE_TRAINER_H 
#define _METIS_PLATFORM_SLAVE_TRAINER_H 


#include <string>
#include <vector>
using namespace std; 
#include "Pattern.h"
using namespace metis_nn;  
#include "SubPerceptron.h"
#include "SubMLP.h"


namespace metis_plat
{

class SlaveTrainer
{
public:
	SlaveTrainer(); 
	virtual ~SlaveTrainer(); 

	bool PushPatt(const char* sPattStr); 
	int32_t PattCnt(); 

	bool SetModel(const char* sModelStr); 
	string GetModel(); 

	void PattShuffle(); 
	void ModelOnceUpdate(const int32_t nBatchCnt, const double dLearningRate, const ERegula eRegula);
	void ModelLoopUpdate(const int32_t nBatchCnt, const double dLearningRate, const ERegula eRegula);
	double GetAvgError(); 

	void ReleaseModel(); 
	void ReleasePatts(); 

	void UpdateCancel(); 
private: 
	SubPerceptron* m_pSubPerceptron; 
	SubMLP* m_pSubMLP; 

	int32_t m_nOff; 
	double m_dAvgError; 

	vector<Pattern*> m_vtrPatts; 
};

}


#endif /* _METIS_PLATFORM_SLAVE_TRAINER_H */

