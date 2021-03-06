#include "Model_MLP.h"
using namespace metis_pred;
#include "Activation.h"
#include "StringArray.h"
using namespace metis_uti;
#include <fstream>
using namespace std; 
#include <math.h>
#include <string.h>


Model_MLP::Model_MLP() : Model(_MODEL_MLP)
{
}


Model_MLP::~Model_MLP()
{
}



bool Model_MLP::Load(const char* sModelFile)
{
	return m_mlp.Load(sModelFile) == _METIS_NN_SUCCESS; 
}


bool Model_MLP::Save(const char* sModelFile)
{
	return m_mlp.Save(sModelFile) == _METIS_NN_SUCCESS; 
}


string Model_MLP::ToString()
{
	return m_mlp.ConvToModelString(); 
}


bool Model_MLP::FromString(const char* sStr)
{
	return m_mlp.SetByModelString(sStr); 
}


double Model_MLP::Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget, const bool bOutliersCheck)
{
	if(nTarget < 0 || nTarget >= N_Output())
		return 0.0; 
	int32_t x_len = N_Input(); 
	double* x = new double[x_len];
	memset(x, 0, x_len * sizeof(double));  
	for(size_t k = 0; k < vtrFeat.size(); k++) 
	{
		if(vtrFeat[k].first < x_len)
		{
			if(bOutliersCheck)
				x[vtrFeat[k].first] = fabs(vtrFeat[k].second) < 100.0 ? vtrFeat[k].second : 0.0;
			else
				x[vtrFeat[k].first] = vtrFeat[k].second; 
		}
	}
	double pred = Predict(x, x_len, nTarget); 
	delete x; 
	return pred; 
}


double Model_MLP::Predict(const double* x, const int32_t nLen, const int32_t nTarget)
{
	if(nTarget < 0 || nTarget >= N_Output())
		return 0.0; 
	int32_t y_len = N_Output(); 
	double* y = new double[y_len]; 
	double pred = 0.0; 
	if(m_mlp.Predict(y, y_len, x, nLen) == _METIS_NN_SUCCESS)
		pred = y[nTarget];
	delete y; 
	return pred;  
}


int32_t Model_MLP::N_Input()
{
	const ArchParams* p_arch_params = m_mlp.GetArchParams(); 
	if(!p_arch_params)
		return 0.0; 
	return p_arch_params->input; 
}


int32_t Model_MLP::N_Output()
{
	const ArchParams* p_arch_params = m_mlp.GetArchParams(); 
	if(!p_arch_params)
		return 0.0; 
	return p_arch_params->output; 
}


