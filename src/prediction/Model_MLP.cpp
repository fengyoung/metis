#include <fstream>
#include <iostream>
using namespace std; 
#include "StringArray.h"
using namespace metis_uti; 
#include "Activation.h"
using namespace metis_nn; 
#include "Model_MLP.h"
using namespace metis_pred; 


Model_MLP::Model_MLP() : Model(_MODEL_MLP)
{
	m_whs = NULL; 
}


Model_MLP::~Model_MLP()
{
	Release(); 
}


bool Model_MLP::Load(const char* sModelFile)
{
	if(!sModelFile)
		return false; 

	ifstream ifs(sModelFile);  
	if(!ifs.is_open())
		return false;

	string str; 
	int32_t idx, hl = 0; 

	std::getline(ifs, str);
	if(str != "** MLP Neural Network **")
	{
		ifs.close(); 
		return false; 
	}
	
	Release(); 

	while(!ifs.eof())
	{	
		std::getline(ifs, str);
		if(str.empty())
			continue; 
		else if(str == "@architecture_params")
		{
			if(!TypeDefs::Read_MLPParamsT(m_paramsMLP, ifs))
			{
				ifs.close(); 
				return false; 
			}
			// create matrices
			hl = (int32_t)m_paramsMLP.vtr_hidden.size(); 
			m_whs = new Matrix[hl]; 
		}
		else if(str.find("@weight_hidden_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 
			if(idx >= hl)
				continue; 
			if(idx == 0)
				m_whs[idx].Create(m_paramsMLP.input, m_paramsMLP.vtr_hidden[idx]); 	
			else	
				m_whs[idx].Create(m_paramsMLP.vtr_hidden[idx-1], m_paramsMLP.vtr_hidden[idx]); 	
			if(!Matrix::Read_Matrix(m_whs[idx], ifs))
			{
				ifs.close(); 
				return false; 
			}	
		}
		else if(str == "@weight_output")
		{
			m_wo.Create(m_paramsMLP.vtr_hidden[hl-1], m_paramsMLP.output); 
			if(!Matrix::Read_Matrix(m_wo, ifs))
			{
				ifs.close(); 
				return false; 
			}	
		}
	}	

	ifs.close(); 
	return true; 
}


bool Model_MLP::Save(const char* sModelFile)
{
	if(!sModelFile)
		return false; 
	if(!m_whs || m_wo.IsNull())
		return false; 

	ofstream ofs(sModelFile); 
	if(!ofs.is_open())
		return false; 

	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();	// number of hidden layers

	ofs<<"** MLP Neural Network **"<<endl; 
	ofs<<endl;

	// save architecture parameters of RBM
	ofs<<"@architecture_params"<<endl; 
	TypeDefs::Print_MLPParamsT(ofs, m_paramsMLP); 
	ofs<<endl; 

	// save transtorm matrix
	for(int32_t h = 0; h < hl; h++) 
	{
		ofs<<"@weight_hidden_"<<h<<endl; 
		Matrix::Print_Matrix(ofs, m_whs[h]);
		ofs<<endl; 
	}
	ofs<<"@weight_output"<<endl; 
	Matrix::Print_Matrix(ofs, m_wo); 
	ofs<<endl; 

	ofs.close(); 
	return true; 
}


string Model_MLP::ToString()
{
	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();    // number of hidden layers	
	char stmp[32];

	string str("mlpnn");
	str += "|@ap:"; 
	str += TypeDefs::ToString_MLPParamsT(m_paramsMLP); 

	for(int32_t h = 0; h < hl; h++) 	
	{
		sprintf(stmp, "|@wh_%d:", h); 
		str += stmp; 	
		str += m_whs[h].ToString(); 
	}
	str += "|@wo:";
	str += m_wo.ToString(); 

	return str; 
}


bool Model_MLP::FromString(const char* sStr)
{
	if(!sStr)
		return false; 

	StringArray array(sStr, "|"); 
	if(array.GetString(0) != "mlpnn")	
		return false; 	
	if(array.Count() < 4)
		return false; 

	Release(); 
	int32_t idx, hl = 0; 

	for(int32_t i = 1; i < array.Count(); i++)
	{
		StringArray ar(array.GetString(i).c_str(), ":"); 
		if(ar.GetString(0) == "@ap")
		{
			if(!TypeDefs::FromString_MLPParamsT(m_paramsMLP, ar.GetString(1).c_str()))
				return false; 
			// create matrices
			hl = (int32_t)m_paramsMLP.vtr_hidden.size(); 
			m_whs = new Matrix[hl]; 
		}
		else if(ar.GetString(0).find("@wh_") == 0)
		{
			StringArray ar_idx(ar.GetString(0).c_str(), "_"); 
			sscanf(ar_idx.GetString(1).c_str(), "%d", &idx); 
			if(idx >= hl)
				continue; 
			if(idx == 0)
				m_whs[idx].Create(m_paramsMLP.input, m_paramsMLP.vtr_hidden[idx]); 	
			else	
				m_whs[idx].Create(m_paramsMLP.vtr_hidden[idx-1], m_paramsMLP.vtr_hidden[idx]); 	
			if(!m_whs[idx].FromString(ar.GetString(1).c_str()))
				return false; 
		}
		else if(ar.GetString(0) == "@wo")
		{
			m_wo.Create(m_paramsMLP.vtr_hidden[hl-1], m_paramsMLP.output); 
			if(!m_wo.FromString(ar.GetString(1).c_str()))
				return false; 	
		}
	}

	return true; 
}


void Model_MLP::Release()
{
	if(m_whs)
	{
		delete [] m_whs; 
		m_whs = NULL; 
	}
	m_wo.Release(); 
}


double Model_MLP::Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget)
{
	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();	// number of hidden layers
	
	double** ahs = new double*[hl];
	for(int32_t h = 0; h < hl; h++)
		ahs[h] = new double[m_paramsMLP.vtr_hidden[h]];  

	// activate hiddens layer by layer
	for(int32_t h = 0; h < hl; h++)
	{
		if(h == 0)
			ActivateHiddenLowest(ahs[h], m_paramsMLP.vtr_hidden[h], vtrFeat, m_paramsMLP.act_hidden, &(m_whs[h])); 
		else	
			ActivateHidden(ahs[h], m_paramsMLP.vtr_hidden[h], ahs[h-1], m_paramsMLP.vtr_hidden[h-1], m_paramsMLP.act_hidden, &(m_whs[h])); 
	}

	// calculate output value
	double pred = ActivateOutput(ahs[hl-1], m_paramsMLP.vtr_hidden[hl-1], m_paramsMLP.act_output, &m_wo, nTarget);  

	for(int32_t h = 0; h < hl; h++)
		delete [] ahs[h];
	delete [] ahs; 
		
	return pred; 
}


bool Model_MLP::CombineWith(Model* pModel, const double w0, const double w1)
{
	if(!pModel)
		return false; 
	if(pModel->GetType() != m_modelType)
		return false; 
	Model_MLP* p_model = (Model_MLP*)pModel; 
	if(!TypeDefs::IsEqual_MLPParamsT(m_paramsMLP, p_model->m_paramsMLP))
		return false; 

	for(int32_t k = 0; k < (int32_t)m_paramsMLP.vtr_hidden.size(); k++) 
	{
		if(!m_whs[k].CombineWith(p_model->m_whs[k], w0, w1))
			return false;
	}

	if(!m_wo.CombineWith(p_model->m_wo, w0, w1))
		return false;

	return true; 
}


void Model_MLP::ActivateHiddenLowest(double* up_a, const int32_t up_size, vector<pair<int32_t,double> >& vtrFeat, 
		const EActType up_act_type, Matrix* p_w)
{
	int32_t i; 
	for(int32_t j = 0; j < up_size; j++) 
	{
		up_a[j] = 0.0;
		for(int32_t k = 0; k < (int32_t)vtrFeat.size(); k++) 
		{
			i = vtrFeat[k].first;
			if(i >= p_w->Rows() - 1)
				continue; 
			up_a[j] += vtrFeat[k].second * (*p_w)[i][j]; 
		}
		up_a[j] += (*p_w)[p_w->Rows()-1][j]; 
		up_a[j] = Activation::Activate(up_a[j], up_act_type); 
	}
}


void Model_MLP::ActivateHidden(double* up_a, const int32_t up_size, const double* low_a, const int32_t low_size, 
		const EActType up_act_type, Matrix* p_w)
{
	for(int32_t j = 0; j < up_size; j++) 
	{
		up_a[j] = 0.0; 
		for(int32_t i = 0; i < low_size; i++) 
			up_a[j] += low_a[i] * (*p_w)[i][j];
		up_a[j] = Activation::Activate(up_a[j], up_act_type);
	}
}


double Model_MLP::ActivateOutput(const double* low_a, const int32_t low_size, 
		const EActType up_act_type, Matrix* p_w, const int32_t nTarget)
{
	double sum = 0.0; 
	for(int32_t i = 0; i < low_size; i++) 
		sum += low_a[i] * (*p_w)[i][nTarget];
	return Activation::Activate(sum, up_act_type); 
}



