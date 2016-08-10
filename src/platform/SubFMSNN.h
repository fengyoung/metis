#ifndef _METIS_PLATFORM_SUB_FMSNN_H 
#define _METIS_PLATFORM_SUB_FMSNN_H 

#include "FMSNN.h"
using namespace metis_nn;


namespace metis_plat
{

class SubFMSNN : public FMSNN 
{
public: 
	SubFMSNN(); 
	virtual ~SubFMSNN(); 

	double OnceUpdate(vector<Pattern*>& vtrPatts, int32_t& nOff, const int32_t nBatchCnt, 
		const double dLearningRate, const ERegula eRegula);

	double LoopUpdate(vector<Pattern*>& vtrPatts, const int32_t nBatchCnt, 
		const double dLearningRate, const ERegula eRegula);
	
	void UpdateCancel(); 

private: 
	bool m_bUpdateCancel; 
};

}


#endif /* _METIS_PLATFORM_SUB_FMSNN_H */

