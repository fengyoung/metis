#include "SlaveTrainer.h"
using namespace metis_plat; 
#include <algorithm>
using namespace std; 


SlaveTrainer::SlaveTrainer() : m_pSubPerceptron(NULL), m_pSubFM(NULL), m_pSubMLP(NULL), m_pSubFMSNN(NULL), m_nOff(0), m_dAvgError(0.0)
{
}


SlaveTrainer::~SlaveTrainer()
{
	ReleaseModel(); 
	ReleasePatts(); 
}


bool SlaveTrainer::PushPatt_inString(const char* sPattStr)
{
	Pattern* ppatt = new Pattern();
	if(!ppatt->FromString(sPattStr))
	{
		delete ppatt; 
		return false; 
	}
	m_vtrPatts.push_back(ppatt); 
	return true; 
}


bool SlaveTrainer::PushPatt_inStream(const char* sPattStream, const int32_t nLen)
{
	Pattern* ppatt = new Pattern(); 
	if(!ppatt->FromStream(sPattStream, nLen))
	{
		delete ppatt; 
		return false; 
	}
	m_vtrPatts.push_back(ppatt); 
	return true; 
}


bool SlaveTrainer::SetModel(const char* sModelStr)
{
	ReleaseModel(); 

	m_pSubPerceptron = new SubPerceptron(); 
	if(m_pSubPerceptron->SetByModelString(sModelStr))
		return true; 
	delete m_pSubPerceptron; 
	m_pSubPerceptron = NULL; 
	
	m_pSubFM = new SubFM(); 
	if(m_pSubFM->SetByModelString(sModelStr))
		return true; 
	delete m_pSubFM; 
	m_pSubFM = NULL; 

	m_pSubMLP = new SubMLP(); 
	if(m_pSubMLP->SetByModelString(sModelStr))
		return true; 
	delete m_pSubMLP; 
	m_pSubMLP = NULL; 
	
	m_pSubFMSNN = new SubFMSNN(); 
	if(m_pSubFMSNN->SetByModelString(sModelStr))
		return true; 
	delete m_pSubFMSNN; 
	m_pSubFMSNN = NULL; 

	return false; 
}


string SlaveTrainer::GetModel()
{
	if(m_pSubPerceptron)
		return m_pSubPerceptron->ConvToModelString(); 	
	else if(m_pSubFM)
		return m_pSubFM->ConvToModelString(); 	
	else if(m_pSubMLP)
		return m_pSubMLP->ConvToModelString(); 	
	else if(m_pSubFMSNN)
		return m_pSubFMSNN->ConvToModelString(); 	
	else
		return string(""); 
}


int32_t SlaveTrainer::PattCnt()
{
	return (int32_t)m_vtrPatts.size(); 
}


void SlaveTrainer::PattShuffle()
{
	random_shuffle(m_vtrPatts.begin(), m_vtrPatts.end()); 
}


void SlaveTrainer::ModelOnceUpdate(const int32_t nBatchCnt, const double dLearningRate, const ERegula eRegula)
{
	if(m_pSubPerceptron)
		m_dAvgError = m_pSubPerceptron->OnceUpdate(m_vtrPatts, m_nOff, nBatchCnt, dLearningRate, eRegula);
	else if(m_pSubFM)
		m_dAvgError = m_pSubFM->OnceUpdate(m_vtrPatts, m_nOff, nBatchCnt, dLearningRate, eRegula);
	else if(m_pSubMLP)
		m_dAvgError = m_pSubMLP->OnceUpdate(m_vtrPatts, m_nOff, nBatchCnt, dLearningRate, eRegula);
	else if(m_pSubFMSNN)
		m_dAvgError = m_pSubFMSNN->OnceUpdate(m_vtrPatts, m_nOff, nBatchCnt, dLearningRate, eRegula);
}


void SlaveTrainer::ModelLoopUpdate(const int32_t nBatchCnt, const double dLearningRate, const ERegula eRegula)
{
	if(m_pSubPerceptron)
		m_dAvgError = m_pSubPerceptron->LoopUpdate(m_vtrPatts, nBatchCnt, dLearningRate, eRegula);
	else if(m_pSubFM)
		m_dAvgError = m_pSubFM->LoopUpdate(m_vtrPatts, nBatchCnt, dLearningRate, eRegula);
	else if(m_pSubMLP)
		m_dAvgError = m_pSubMLP->LoopUpdate(m_vtrPatts, nBatchCnt, dLearningRate, eRegula);
	else if(m_pSubFMSNN)
		m_dAvgError = m_pSubFMSNN->LoopUpdate(m_vtrPatts, nBatchCnt, dLearningRate, eRegula);
}


double SlaveTrainer::GetAvgError()
{
	return m_dAvgError; 
}


void SlaveTrainer::ReleaseModel()
{
	if(m_pSubPerceptron)
	{
		delete m_pSubPerceptron; 
		m_pSubPerceptron = NULL; 
	}
	if(m_pSubFM)
	{
		delete m_pSubFM; 
		m_pSubFM = NULL; 
	}
	if(m_pSubMLP)
	{
		delete m_pSubMLP; 
		m_pSubMLP = NULL; 
	}
	if(m_pSubFMSNN)
	{
		delete m_pSubFMSNN; 
		m_pSubFMSNN = NULL; 
	}
	m_dAvgError = 0.0; 
}


void SlaveTrainer::ReleasePatts()
{
	for(size_t i = 0; i < m_vtrPatts.size(); i++)
		delete m_vtrPatts[i]; 
	m_vtrPatts.clear(); 
	m_nOff = 0; 
}


void SlaveTrainer::UpdateCancel()
{
	if(m_pSubPerceptron)
		m_pSubPerceptron->UpdateCancel(); 
	if(m_pSubFM)
		m_pSubFM->UpdateCancel(); 
	if(m_pSubMLP)
		m_pSubMLP->UpdateCancel(); 
	if(m_pSubFMSNN)
		m_pSubFMSNN->UpdateCancel(); 
}


