#include <fstream>
#include <iostream>
using namespace std; 
#include "StringArray.h"
using namespace metis_uti; 
#include "Activation.h"
using namespace metis_nn; 
#include "Model_FMSNN.h"
using namespace metis_pred; 


Model_FMSNN::Model_FMSNN() : Model(_MODEL_FMSNN)
{
	m_vfs = NULL; 
	m_whs = NULL; 
}


Model_FMSNN::~Model_FMSNN()
{
	Release(); 
}


bool Model_FMSNN::Load(const char* sModelFile)
{
	ifstream ifs(sModelFile);  
	if(!ifs.is_open())
		return false; 
	Release(); 

	string str; 
	int32_t idx; 

	std::getline(ifs, str); 
	if(str != "** FM Supported Neural Network **")
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
			if(!TypeDefs::Read_FMSNNParamsT(m_paramsFMSNN, ifs))
				return false; 
			//m_wf.Create(m_paramsFMSNN.input, m_paramsFMSNN.fm_layer - m_paramsFMSNN.fm_inter); 
			int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();    // number of hidden layers	
			int32_t fm_layer = m_paramsFMSNN.fm_inter + m_paramsFMSNN.input; 
			m_vfs = new Matrix[m_paramsFMSNN.fm_inter];
			for(int32_t j = 0; j < m_paramsFMSNN.fm_inter; j++) 
				m_vfs[j].Create(m_paramsFMSNN.input - 1, m_paramsFMSNN.fm_k); 
			m_whs = new Matrix[hl];
			for(int32_t h = 0; h < hl; h++) 
			{
				if(h == 0)
					m_whs[h].Create(fm_layer, m_paramsFMSNN.vtr_hidden[h]);
				else
					m_whs[h].Create(m_paramsFMSNN.vtr_hidden[h-1], m_paramsFMSNN.vtr_hidden[h]);
			}
			m_wo.Create(m_paramsFMSNN.vtr_hidden[hl-1], m_paramsFMSNN.output); 
		}
	//	else if(str == "@weight_fmlayer")
	//	{
	//		if(!Matrix::Read_Matrix(m_wf, ifs))
	//			return false; 
	//	}
		else if(str.find("@interaction_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 
			if(!Matrix::Read_Matrix(m_vfs[idx], ifs))
				return false; 
		}
		else if(str.find("@weight_hidden_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 
			if(!Matrix::Read_Matrix(m_whs[idx], ifs))
				return false; 
		}
		else if(str == "@weight_output")
		{
			if(!Matrix::Read_Matrix(m_wo, ifs))
				return false; 
		}
	}

	ifs.close(); 
	return true; 
}


bool Model_FMSNN::Save(const char* sModelFile)
{
	//if(m_wf.IsNull() || !m_vfs || !m_whs || m_wo.IsNull())
	if(!m_vfs || !m_whs || m_wo.IsNull())
		return false; 
	ofstream ofs(sModelFile); 
	if(!ofs.is_open())
		return false; 

	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();	// number of hidden layers

	ofs<<"** FM Supported Neural Network **"<<endl; 
	ofs<<endl;

	// save architecture parameters of FMSNN 
	ofs<<"@architecture_params"<<endl; 
	TypeDefs::Print_FMSNNParamsT(ofs, m_paramsFMSNN); 
	ofs<<endl; 

/*
	// save weight matrix of FM layer
	ofs<<"@weight_fmlayer"<<endl; 
	Matrix::Print_Matrix(ofs, m_wf);
	ofs<<endl; 
*/
	// save interaction matrix of FM layer
	for(int32_t j = 0; j < m_paramsFMSNN.fm_inter; j++) 
	{
		ofs<<"@interaction_"<<j<<endl; 
		Matrix::Print_Matrix(ofs, m_vfs[j]);
		ofs<<endl; 
	}

	// save transtorm matrices of hidden layers
	for(int32_t h = 0; h < hl; h++) 
	{
		ofs<<"@weight_hidden_"<<h<<endl; 
		Matrix::Print_Matrix(ofs, m_whs[h]);
		ofs<<endl; 
	}

	// save transtorm matrix of output layer
	ofs<<"@weight_output"<<endl; 
	Matrix::Print_Matrix(ofs, m_wo); 
	ofs<<endl; 

	ofs.close(); 
	return true; 
}


string Model_FMSNN::ToString()
{
	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();    // number of hidden layers	
	char stmp[32];

	string str("fmsnn");
	str += "|@ap:"; 
	str += TypeDefs::ToString_FMSNNParamsT(m_paramsFMSNN); 

//	str += "|@wf:";
//	str += m_wf.ToString(); 

	for(int32_t j = 0; j < m_paramsFMSNN.fm_inter; j++) 
	{
		sprintf(stmp, "|@vf_%d:", j); 
		str += stmp; 	
		str += m_vfs[j].ToString(); 
	}

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


bool Model_FMSNN::FromString(const char* sModelStr)
{
	if(!sModelStr)
		return false; 

	StringArray array(sModelStr, "|"); 
	if(array.GetString(0) != "fmsnn")	
		return false; 	
	if(array.Count() < 5)
		return false; 

	Release(); 
	int32_t idx, hl = 0; 

	for(int32_t i = 1; i < array.Count(); i++)
	{
		StringArray ar(array.GetString(i).c_str(), ":"); 
		if(ar.GetString(0) == "@ap")
		{
			if(!TypeDefs::FromString_FMSNNParamsT(m_paramsFMSNN, ar.GetString(1).c_str()))
				return false; 
			// create matrices
			hl = (int32_t)m_paramsFMSNN.vtr_hidden.size(); 
			m_whs = new Matrix[hl]; 
			m_vfs = new Matrix[m_paramsFMSNN.fm_inter]; 
		}
		/*
		else if(ar.GetString(0) == "@wf")
		{
			if(!m_wf.FromString(ar.GetString(1).c_str()))
				return false; 	
		}
		*/
		else if(ar.GetString(0).find("@vf_") == 0)
		{
			StringArray ar_idx(ar.GetString(0).c_str(), "_"); 
			sscanf(ar_idx.GetString(1).c_str(), "%d", &idx); 
			if(idx >= m_paramsFMSNN.fm_inter)
				continue; 
			if(!m_vfs[idx].FromString(ar.GetString(1).c_str()))
				return false; 
		}
		else if(ar.GetString(0).find("@wh_") == 0)
		{
			StringArray ar_idx(ar.GetString(0).c_str(), "_"); 
			sscanf(ar_idx.GetString(1).c_str(), "%d", &idx); 
			if(idx >= hl)
				continue; 
			if(!m_whs[idx].FromString(ar.GetString(1).c_str()))
				return false; 
		}
		else if(ar.GetString(0) == "@wo")
		{
			if(!m_wo.FromString(ar.GetString(1).c_str()))
				return false; 	
		}
	}

	return true; 
}


void Model_FMSNN::Release()
{
	//m_wf.Release(); 
	if(m_vfs)
	{
		delete [] m_vfs; 
		m_vfs = NULL; 
	}
	if(m_whs)
	{
		delete [] m_whs; 
		m_whs = NULL; 
	}
	m_wo.Release(); 
}


double Model_FMSNN::Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget)
{
	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();	// number of hidden layers
	int32_t fm_layer = m_paramsFMSNN.fm_inter + m_paramsFMSNN.input; 
	double* af = new double[fm_layer]; 
	double** ahs = new double*[hl];
	for(int32_t h = 0; h < hl; h++)
		ahs[h] = new double[m_paramsFMSNN.vtr_hidden[h]];  

	ActivateFMLayer(af, m_paramsFMSNN.fm_inter, vtrFeat, m_vfs, m_paramsFMSNN.fm_k, m_paramsFMSNN.act_fm_layer); 

	for(int32_t h = 0; h < hl; h++) 
	{
		if(h == 0)
			ActivateHidden(ahs[h], m_paramsFMSNN.vtr_hidden[h], af, fm_layer, m_whs[h], m_paramsFMSNN.act_hidden);
		else
			ActivateHidden(ahs[h], m_paramsFMSNN.vtr_hidden[h], ahs[h-1], m_paramsFMSNN.vtr_hidden[h-1], m_whs[h], m_paramsFMSNN.act_hidden);
	}

	double pred = ActivateOutput(ahs[hl-1], m_paramsFMSNN.vtr_hidden[hl-1], m_wo, m_paramsFMSNN.act_output, nTarget); 

	delete [] af; 	
	for(int32_t h = 0; h < hl; h++)
		delete [] ahs[h];
	delete [] ahs; 

	return pred; 
}



bool Model_FMSNN::CombineWith(Model* pModel, const double w0, const double w1)
{
	if(!pModel)
		return false; 
	if(pModel->GetType() != m_modelType)
		return false; 
	Model_FMSNN* p_model = (Model_FMSNN*)pModel; 

	if(!TypeDefs::IsEqual_FMSNNParamsT(m_paramsFMSNN, p_model->m_paramsFMSNN))
		return false; 

/*
	if(!m_wf.CombineWith(p_model->m_wf, w0, w1))
		return false; 
*/

	for(int32_t j = 0; j < m_paramsFMSNN.fm_inter; j++) 
	{
		if(!m_vfs[j].CombineWith(p_model->m_vfs[j], w0, w1))
			return false; 	
	}

	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();    // number of hidden layers	
	for(int32_t h = 0; h < hl; h++) 
	{
		if(!m_whs[h].CombineWith(p_model->m_whs[h], w0, w1))
			return false; 	
	}

	if(!m_wo.CombineWith(p_model->m_wo, w0, w1))
		return false; 

	return true; 
}


void Model_FMSNN::ActivateFMLayer(double* af, const int32_t inter_size, vector<pair<int32_t,double> >& vtrFeat, 
		Matrix* vfs, const int32_t fm_k, const EActType act_type)
{
	double sum, inter_sum1, inter_sum2; 
	int32_t i; 

	// interactions 		
	for(int32_t j = 0; j < inter_size; j++) 
	{
		sum = 0.0; 
		for(int32_t k = 0; k < fm_k; k++) 
		{
			inter_sum1 = 0.0; 
			inter_sum2 = 0.0; 
			for(size_t s = 0; s < vtrFeat.size(); s++) 
			{
				i = vtrFeat[s].first;
				if(i >= vfs[j].Rows() - 1)
					continue; 
				inter_sum1 += vtrFeat[s].second * vfs[j][i][k]; 
				inter_sum2 += vtrFeat[s].second * vfs[j][i][k] * vtrFeat[s].second * vfs[j][i][k]; 
			}	
			sum += (inter_sum1 * inter_sum1 - inter_sum2) / 2.0;     	
		}
		af[j] = Activation::Activate(sum, act_type); 
	}	

	for(size_t s = 0; s < vtrFeat.size(); s++) 
	{
		i = vtrFeat[s].first;
		if(i >= vfs[0].Rows() - 1)
			continue; 
		af[i + inter_size] = vtrFeat[s].second; 	
	}
}


void Model_FMSNN::ActivateHidden(double* up_a, const int32_t up_size, const double* low_a, const int32_t low_size, 
		Matrix& w, const EActType up_act_type)
{
	double e = 0.0; 

	for(int32_t j = 0; j < up_size; j++) 
	{
		up_a[j] = 0.0; 
		for(int32_t i = 0; i < low_size; i++) 
			up_a[j] += low_a[i] * w[i][j];
		up_a[j] = Activation::Activate(up_a[j], up_act_type);

		if(up_act_type == _ACT_SOFTMAX)
			e += up_a[j]; 	
	}
	if(up_act_type == _ACT_SOFTMAX)
	{
		for(int32_t j = 0; j < up_size; j++) 
			up_a[j] /= e; 	
	}
}


double Model_FMSNN::ActivateOutput(const double* low_a, const int32_t low_size, Matrix& w, const EActType up_act_type, 
		const int32_t nTarget)
{
	if(up_act_type== _ACT_SOFTMAX)
	{
		double e = 0.0; 
		double* y = new double[w.Cols()];  
		for(int32_t j = 0; j < w.Cols(); j++) 
		{
			y[j] = 0.0; 
			for(int32_t i = 0; i < low_size; i++) 
				y[j] += low_a[i] * w[i][j];
			y[j] = Activation::Activate(y[j], up_act_type);
			e += y[j]; 
		}
		for(int32_t j = 0; j < w.Cols(); j++) 
			y[j] /= e;
		double pred = y[nTarget]; 
		delete [] y;
		return pred; 
	}
	else
	{
		double sum = 0.0; 
		for(int32_t i = 0; i < low_size; i++) 
			sum += low_a[i] * w[i][nTarget];
		return Activation::Activate(sum, up_act_type); 
	}
}



