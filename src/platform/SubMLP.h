#ifndef _METIS_PLATFORM_SUB_MLP_H 
#define _METIS_PLATFORM_SUB_MLP_H 

#include "MLP.h"
using namespace metis_nn;


namespace metis_plat
{

class SubMLP : public MLP
{
public: 
	SubMLP(); 
	virtual ~SubMLP(); 

	double OnceUpdate(vector<Pattern*>& vtrPatts, int32_t& nOff, const int32_t nBatchCnt, 
		const double dLearningRate, const ERegula eRegula);
	
	double LoopUpdate(vector<Pattern*>& vtrPatts, const int32_t nBatchCnt, 
		const double dLearningRate, const ERegula eRegula);

	void UpdateCancel(); 

private: 
	bool m_bUpdateCancel; 
};

}


#endif /* _METIS_PLATFORM_SUB_MLP_H */

