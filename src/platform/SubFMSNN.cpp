#include "SubFMSNN.h"
using namespace metis_plat; 
#include <math.h>


SubFMSNN::SubFMSNN() : FMSNN(), m_bUpdateCancel(false)
{
}


SubFMSNN::~SubFMSNN()
{
}


double SubFMSNN::OnceUpdate(vector<Pattern*>& vtrPatts, int32_t& nOff, const int32_t nBatchCnt, 
		const double dLearningRate, const ERegula eRegula)
{
	m_bUpdateCancel = false; 
	// create assistant variables for training
	CreateAssistant();

	double error = 0.0; 	
	int32_t patt_cnt = 0; 

	while(patt_cnt < nBatchCnt)
	{
		if(m_bUpdateCancel)
			break; 
		// forward & backward phase
		FeedForward(vtrPatts[nOff]->m_x, vtrPatts[nOff]->m_nXCnt); 
		error += BackPropagate(vtrPatts[nOff]->m_y, vtrPatts[nOff]->m_nYCnt);
		// offset
		nOff++; 
		if(nOff >= (int32_t)vtrPatts.size())
			nOff = 0; 
		// counter	
		patt_cnt++; 	
	}
	// update the model 
	ModelUpdate(dLearningRate, eRegula, patt_cnt); 
	
	return error / (double)(patt_cnt); 
}


double SubFMSNN::LoopUpdate(vector<Pattern*>& vtrPatts, const int32_t nBatchCnt, 
		const double dLearningRate, const ERegula eRegula)
{
	m_bUpdateCancel = false; 
	// create assistant variables for training
	CreateAssistant();

	double error = 0.0; 	
	int32_t patt_cnt = 0; 

	for(int32_t t = 0; t < (int32_t)vtrPatts.size(); t++) 
	{
		if(m_bUpdateCancel)
			break; 
		// forward & backward phase
		FeedForward(vtrPatts[t]->m_x, vtrPatts[t]->m_nXCnt); 
		error += BackPropagate(vtrPatts[t]->m_y, vtrPatts[t]->m_nYCnt); 
		patt_cnt++; 

		if(nBatchCnt > 0 && patt_cnt >= nBatchCnt)
		{
			ModelUpdate(dLearningRate, eRegula, patt_cnt); 
			patt_cnt = 0; 	
		}
	}
	if(patt_cnt > 0)
		ModelUpdate(dLearningRate, eRegula, patt_cnt); 
	return error / (double)vtrPatts.size(); 
}


void SubFMSNN::UpdateCancel()
{
	m_bUpdateCancel = true; 
}


