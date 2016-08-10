#include <fstream>
#include <iostream>
using namespace std; 
#include "StringArray.h"
using namespace metis_uti; 
#include "Activation.h"
using namespace metis_nn; 
#include "Model_Perceptron.h"
using namespace metis_pred; 


Model_Perceptron::Model_Perceptron() : Model(_MODEL_PERCEPTRON)
{
}


Model_Perceptron::~Model_Perceptron()
{
	Release(); 
}


bool Model_Perceptron::Load(const char* sModelFile) 
{
	if(!sModelFile)
		return false; 

	ifstream ifs(sModelFile);  
	if(!ifs.is_open())
		return false; 

	Release(); 
	string str; 
	
	std::getline(ifs, str);
	if(str != "** Perceptron **")
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
			if(!TypeDefs::Read_PerceptronParamsT(m_paramsPerceptron, ifs))
			{
				ifs.close(); 
				return false; 
			}
		}
		else if(str == "@weight")
		{
			m_wo.Create(m_paramsPerceptron.input, m_paramsPerceptron.output); 
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


bool Model_Perceptron::Save(const char* sModelFile)
{
	if(!sModelFile)
		return false; 
	if(m_wo.IsNull())
		return false; 

	ofstream ofs(sModelFile); 

	ofs<<"** Perceptron **"<<endl; 
	ofs<<endl;

	// save architecture parameters of RBM
	ofs<<"@architecture_params"<<endl; 
	TypeDefs::Print_PerceptronParamsT(ofs, m_paramsPerceptron); 
	ofs<<endl; 

	// save transtorm matrix
	ofs<<"@weight"<<endl; 
	Matrix::Print_Matrix(ofs, m_wo);
	ofs<<endl; 

	ofs.close(); 
	return true; 
}


string Model_Perceptron::ToString()
{
	string str("perceptron");
	str += "|@ap:"; 
	str += TypeDefs::ToString_PerceptronParamsT(m_paramsPerceptron);  
	str += "|@w:"; 
	str += m_wo.ToString(); 
	return str; 
}


bool Model_Perceptron::FromString(const char* sStr)
{
	if(!sStr)
		return false;

	StringArray array(sStr, "|"); 
	if(array.GetString(0) != "perceptron")	
		return false; 	
	if(array.Count() < 3)
		return false; 

	Release(); 
	
	for(int32_t i = 1; i < array.Count(); i++) 
	{
		StringArray ar(array.GetString(i).c_str(), ":"); 
		if(ar.GetString(0) == "@ap")
		{
			if(!TypeDefs::FromString_PerceptronParamsT(m_paramsPerceptron, ar.GetString(1).c_str()))
				return false; 
		}
		else if(ar.GetString(0) == "@w")
		{
			if(!m_wo.FromString(ar.GetString(1).c_str()))
				return false; 
		}
	}

	return true; 
}


void Model_Perceptron::Release()
{
	m_wo.Release(); 
}


double Model_Perceptron::Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget) 
{
	if(nTarget >= m_paramsPerceptron.output)
		return 0.0; 

	double sum = 0.0; 
	int32_t i; 

	for(int32_t k = 0; k < (int32_t)vtrFeat.size(); k++)
	{
		i = vtrFeat[k].first; 
		if(i >= m_paramsPerceptron.input - 1)
			continue; 
		sum += vtrFeat[k].second * m_wo[i][nTarget]; 
	}	
	sum += m_wo[m_paramsPerceptron.input - 1][nTarget];	// for bias

	return Activation::Activate(sum, m_paramsPerceptron.act_output); 
}


bool Model_Perceptron::CombineWith(Model* pModel, const double w0, const double w1)
{
	if(!pModel)
		return false; 
	if(pModel->GetType() != m_modelType)
		return false; 
	Model_Perceptron* p_model = (Model_Perceptron*)pModel; 
	if(!TypeDefs::IsEqual_PerceptronParamsT(m_paramsPerceptron, p_model->m_paramsPerceptron))
		return false; 

	if(!m_wo.CombineWith(p_model->m_wo, w0, w1))
		return false;

	return true; 
}



