#include <fstream>
#include <iostream>
using namespace std; 
#include "StringArray.h"
using namespace metis_uti; 
#include "Activation.h"
using namespace metis_nn; 
#include "Model_FM.h"
using namespace metis_pred; 


Model_FM::Model_FM() : Model(_MODEL_FM)
{
	m_vo = NULL; 
}


Model_FM::~Model_FM()
{
	Release();
}


bool Model_FM::Load(const char* sModelFile)
{
	if(!sModelFile)
		return false; 

	ifstream ifs(sModelFile);  
	if(!ifs.is_open())
		return false; 

	Release(); 
	string str; 
	int32_t idx; 

	std::getline(ifs, str);
	if(str != "** Factorization Machine **")
	{
		ifs.close(); 
		return false; 
	}

	while(!ifs.eof())
	{
		std::getline(ifs, str);
		if(str.empty())
			continue; 
		else if(str == "@architecture_params")
		{
			if(!TypeDefs::Read_FMParamsT(m_paramsFM, ifs))
			{
				ifs.close(); 
				return false; 
			}
			m_wo.Create(m_paramsFM.input, m_paramsFM.output); 
			m_vo = new Matrix[m_paramsFM.output];
			for(int32_t j = 0; j < m_paramsFM.output; j++) 
				m_vo[j].Create(m_paramsFM.input-1, m_paramsFM.fm_k);
		}
		else if(str == "@weight")
		{
			if(!Matrix::Read_Matrix(m_wo, ifs))
			{
				ifs.close(); 
				return false; 
			}
		}
		else if(str.find("@interaction_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 
			if(!Matrix::Read_Matrix(m_vo[idx], ifs))
				return _METIS_NN_ERROR_MODEL_DATA;
		}
	}

	ifs.close(); 
	return true; 
}


bool Model_FM::Save(const char* sModelFile)
{
	if(m_wo.IsNull() || !m_vo)
		return false; 
	ofstream ofs(sModelFile); 
	if(!ofs.is_open())
		return false; 

	ofs<<"** Factorization Machine **"<<endl; 
	ofs<<endl;

	// save architecture parameters of FM
	ofs<<"@architecture_params"<<endl; 
	TypeDefs::Print_FMParamsT(ofs, m_paramsFM); 
	ofs<<endl; 

	// save transtorm matrix
	ofs<<"@weight"<<endl; 
	Matrix::Print_Matrix(ofs, m_wo);
	ofs<<endl; 

	// save interaction matrices
	for(int32_t j = 0; j < m_paramsFM.output; j++) 
	{
		ofs<<"@interaction_"<<j<<endl; 
		Matrix::Print_Matrix(ofs, m_vo[j]); 
		ofs<<endl; 
	}

	ofs.close(); 
	return _METIS_NN_SUCCESS; 
}


string Model_FM::ToString()
{
	string str("fm");
	str += "|@ap:"; 
	str += TypeDefs::ToString_FMParamsT(m_paramsFM);  
	str += "|@w:"; 
	str += m_wo.ToString();
	char stmp[32]; 
	for(int32_t j = 0; j < m_paramsFM.output; j++)
	{
		sprintf(stmp, "|@inter_%d:", j);
		str += stmp; 	
		str += m_vo[j].ToString(); 
	}
	return str; 
}


bool Model_FM::FromString(const char* sStr)
{
	if(!sStr)
		return false;

	StringArray array(sStr, "|"); 
	if(array.GetString(0) != "fm")	
		return false; 	
	if(array.Count() < 4)
		return false; 

	Release(); 
	int32_t idx; 

	for(int32_t i = 1; i < array.Count(); i++) 
	{
		StringArray ar(array.GetString(i).c_str(), ":"); 
		if(ar.GetString(0) == "@ap")
		{
			if(!TypeDefs::FromString_FMParamsT(m_paramsFM, ar.GetString(1).c_str()))
				return false; 
			m_vo = new Matrix[m_paramsFM.output];
		}
		else if(ar.GetString(0) == "@w")
		{
			if(!m_wo.FromString(ar.GetString(1).c_str()))
				return false; 
		}
		else if(ar.GetString(0).find("@inter_") == 0)
		{
			StringArray ar_idx(ar.GetString(0).c_str(), "_"); 
			sscanf(ar_idx.GetString(1).c_str(), "%d", &idx); 
			if(idx >= m_paramsFM.output)
				continue; 
			if(!m_vo[idx].FromString(ar.GetString(1).c_str()))
				return false; 
		}
	}

	return true; 
}


void Model_FM::Release()
{
	m_wo.Release(); 
	if(m_vo)
	{
		delete [] m_vo; 
		m_vo = NULL; 
	}
}


double Model_FM::Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget)
{
	if(nTarget >= m_paramsFM.output)
		return 0.0; 

	double sum = 0.0, inter_sum1,  inter_sum2, xn; 
	int32_t i; 

	// dim effect
	for(int32_t h = 0; h < (int32_t)vtrFeat.size(); h++)
	{
		i = vtrFeat[h].first; 
		if(i >= m_paramsFM.input - 1)
			continue; 
		sum += vtrFeat[h].second * m_wo[i][nTarget]; 
	}	
	sum += m_wo[m_paramsFM.input - 1][nTarget];	// for bias

	// interaction effect
	for(int32_t k = 0; k < m_paramsFM.fm_k; k++) 
	{
		inter_sum1 = 0.0; 
		inter_sum2 = 0.0; 
		for(int32_t h = 0; h < (int32_t)vtrFeat.size(); h++)
		{
			i = vtrFeat[h].first; 
			if(i >= m_paramsFM.input - 1)
				continue; 
			xn = vtrFeat[h].second; 

			inter_sum1 += m_vo[nTarget][i][k] * xn; 
			inter_sum2 += m_vo[nTarget][i][k] * m_vo[nTarget][i][k] * xn * xn; 
		}
		sum += (inter_sum1 * inter_sum1 - inter_sum2) / 2.0; 
	}

	return Activation::Activate(sum, m_paramsFM.act_output); 
}


bool Model_FM::CombineWith(Model* pModel, const double w0, const double w1)
{
	if(!pModel)
		return false; 
	if(pModel->GetType() != m_modelType)
		return false; 
	Model_FM* p_model = (Model_FM*)pModel; 
	if(!TypeDefs::IsEqual_FMParamsT(m_paramsFM, p_model->m_paramsFM))
		return false; 

	if(!m_wo.CombineWith(p_model->m_wo, w0, w1))
		return false;
	
	for(int32_t j = 0; j < m_paramsFM.output; j++)
	{
		if(!m_vo[j].CombineWith(p_model->m_vo[j], w0, w1))
			return false; 
	}

	return true; 
}


