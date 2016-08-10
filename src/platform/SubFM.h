#ifndef _METIS_PLATFORM_SUB_FM_H 
#define _METIS_PLATFORM_SUB_FM_H 

#include "FM.h"
using namespace metis_nn;


namespace metis_plat
{

class SubFM : public FM 
{
public: 
	SubFM(); 
	virtual ~SubFM(); 

	double OnceUpdate(vector<Pattern*>& vtrPatts, int32_t& nOff, const int32_t nBatchCnt, 
		const double dLearningRate, const ERegula eRegula);

	double LoopUpdate(vector<Pattern*>& vtrPatts, const int32_t nBatchCnt, 
		const double dLearningRate, const ERegula eRegula);
	
	void UpdateCancel(); 

private: 
	bool m_bUpdateCancel; 
};

}


#endif /* _METIS_PLATFORM_SUB_FM_H */

