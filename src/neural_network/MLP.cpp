#include "MLP.h"
#include "Activation.h"
using namespace metis_nn; 
#include "Config.h"
#include "StringArray.h"
using namespace metis_uti; 
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std; 
#include <string.h>
#include <math.h>



MLP::MLP() : NeuralNetwork(_NN_MLP), m_ws(NULL)
{
}


MLP::~MLP() 
{
	Release(); 
}


bool MLP::Init(LearnParams* pLearnParams, ArchParams* pArchParams)
{
	if(!pLearnParams || !pArchParams)
		return false; 
	if(pArchParams->GetType() != _ARCH_PARAMS_MLP)
		return false; 

	if(m_pLearnParams)
		delete m_pLearnParams; 
	m_pLearnParams = new LearnParams(*pLearnParams); 
	if(m_pArchParams)
		delete m_pArchParams; 
	m_pArchParams = new ArchParams_MLP(*((ArchParams_MLP*)pArchParams)); 
	Create(); 

	return true; 
}


bool MLP::InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	if(conf.ValCnt("Model") == 0)
		return false;
	if(conf.GetVal_asString("Model") != "MLP")
		return false;
	if(m_pLearnParams)	
		delete m_pLearnParams; 
	m_pLearnParams = new LearnParams(); 
	if(m_pArchParams)
		delete m_pArchParams; 
	m_pArchParams = new ArchParams_MLP(); 

	if(!m_pArchParams->FromConfig(sConfigFile, nInput, nOutput))
		return false; 
	if(!m_pLearnParams->FromConfig(sConfigFile))
		return false; 

	Create(); 
	return true; 
}


bool MLP::Epoch(double& dAvgLoss, NNAssi** ppAssi, vector<Pattern*>& vtrPatts, const double dLearningRate, const int32_t nStartOff)
{
	if(IsNull() || !ppAssi)
		return false; 
	// shuffle training patterns
	random_shuffle(vtrPatts.begin() + nStartOff, vtrPatts.end());

	dAvgLoss = 0.0; 
	double loss; 
	int32_t patt_cnt = 0; 
	int32_t batch_cnt = m_pLearnParams->batch_size <= 0 ? (int32_t)vtrPatts.size() : m_pLearnParams->batch_size; 
	// one epoch
	for(int32_t t = nStartOff; t < (int32_t)vtrPatts.size(); t++) 
	{
		if(m_bUpdateCancel)
			break; 	
		// forward & backward phase
		FeedForward(ppAssi, vtrPatts[t]->m_x, vtrPatts[t]->m_nXCnt); 
		BackPropagate(loss, ppAssi, vtrPatts[t]->m_x, vtrPatts[t]->m_nXCnt, vtrPatts[t]->m_y, vtrPatts[t]->m_nYCnt); 
		dAvgLoss += loss; 
		patt_cnt++; 

		if(patt_cnt >= batch_cnt)
		{
			ModelUpdate(ppAssi, dLearningRate); 
			patt_cnt = 0; 	
		}
	}
	if(patt_cnt > 0)
	{
		ModelUpdate(ppAssi, dLearningRate); 
		patt_cnt = 0; 	
	}
	dAvgLoss /= (double)((int32_t)vtrPatts.size() - nStartOff); 
	return true; 
}


int32_t MLP::Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len)
{
	if(!y || !x)
		return _METIS_NN_ERROR_INPUT_NULL; 
	if(IsNull())
		return _METIS_NN_ERROR_MODEL_NULL; 
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	if(x_len < 0 || x_len != p_arch_params->input || y_len < 0)
		return _METIS_NN_ERROR_WRONG_LEN; 

	int32_t hl = (int32_t)(p_arch_params->vtr_hiddens.size()); 
	NNAssi** pp_assi = CreateAssi(false); 
	if(!pp_assi)
		return _METIS_NN_ASSI_ERROR; 		

	FeedForward(pp_assi, x, x_len); 

	for(int32_t j = 0; j < y_len; j++)
	{
		if(j < m_pArchParams->output)
			y[j] = pp_assi[hl]->m_ao[j];
		else
			y[j] = 0.0; 
	}

	ReleaseAssi(pp_assi); 
	return _METIS_NN_SUCCESS; 
}


int32_t MLP::Save(const char* sModelFile)
{
	if(!sModelFile)
		return _METIS_NN_ERROR_INPUT_NULL; 
	if(IsNull())
		return _METIS_NN_ERROR_MODEL_NULL; 
	ofstream ofs(sModelFile); 
	if(!ofs.is_open())
		return _METIS_NN_ERROR_FILE_OPEN;

	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams;  
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size(); 

	ofs<<"** Multiple Layers Perceptron **"<<endl; 
	ofs<<endl;

	// save learning parameters
	ofs<<"@learning_params"<<endl; 
	m_pLearnParams->Print(ofs); 
	ofs<<endl; 

	// save architecture parameters of perceptron
	ofs<<"@architecture_params"<<endl; 
	p_arch_params->Print(ofs); 
	ofs<<endl; 

	// save transtorm matrices
	for(int32_t h = 0; h <= hl; h++) 
	{	
		ofs<<"@weight_"<<h<<endl; 
		Matrix::Print_Matrix(ofs, m_ws[h]);
		ofs<<endl; 
	}

	ofs.close(); 
	return _METIS_NN_SUCCESS; 
}


int32_t MLP::Load(const char* sModelFile)
{
	if(!sModelFile)
		return _METIS_NN_ERROR_INPUT_NULL; 
	ifstream ifs(sModelFile); 
	if(!ifs.is_open())
		return _METIS_NN_ERROR_FILE_OPEN; 
	if(m_pLearnParams)	
		delete m_pLearnParams; 
	m_pLearnParams = new LearnParams(); 
	if(m_pArchParams)
		delete m_pArchParams; 
	m_pArchParams = new ArchParams_MLP(); 
	Release(); 

	string str; 
	int32_t h, hl = 0; 

	std::getline(ifs, str); 
	if(str != "** Multiple Layers Perceptron **")
	{
		ifs.close();
		return _METIS_NN_ERROR_NOT_MODEL_FILE;
	}
	while(!ifs.eof())
	{
		std::getline(ifs, str);
		if(str.empty())
			continue; 
		else if(str == "@learning_params")
		{
			if(!m_pLearnParams->Read(ifs))
				return _METIS_NN_ERROR_LERANING_PARAMS;
		}
		else if(str == "@architecture_params")
		{
			if(!m_pArchParams->Read(ifs))
				return _METIS_NN_ERROR_ACH_PARAMS;
			Create(); 
			hl = (int32_t)((ArchParams_MLP*)m_pArchParams)->vtr_hiddens.size(); 	
		}
		else if(str.find("@weight_") == 0)
		{
			StringArray ar_h(str.c_str(), "_"); 
			sscanf(ar_h.GetString(ar_h.Count() - 1).c_str(), "%d", &h);
			if(h > hl)
				continue; 	
			if(!Matrix::Read_Matrix(m_ws[h], ifs))
				return _METIS_NN_ERROR_MODEL_DATA;
		}
	}

	ifs.close(); 
	return _METIS_NN_SUCCESS; 
}


bool MLP::CombineWith(NeuralNetwork* pNN, const double w0, const double w1)
{
	if(!CompArchWith(pNN))
		return false; 
	MLP* p = (MLP*)pNN;
	int32_t hl = (int32_t)((ArchParams_MLP*)m_pArchParams)->vtr_hiddens.size(); 	
	for(int32_t h = 0; h <= hl; h++)
	{
		if(!m_ws[h].CombineWith(p->m_ws[h], w0, w1))
			return false; 
	}
	return true; 
}


void MLP::NumericMultiWith(const double a)
{
	if(IsNull())
		return; 
	int32_t hl = (int32_t)((ArchParams_MLP*)m_pArchParams)->vtr_hiddens.size(); 	
	for(int32_t h = 0; h <= hl; h++)
		m_ws[h].NumericMultiWith(a); 
}


bool MLP::SetByModelString(const char* sModelStr)
{
	if(!sModelStr)
		return false;

	StringArray array(sModelStr, "|"); 
	if(array.GetString(0) != "mlp")	
		return false; 	
	if(array.Count() < 4)
		return false; 

	Release(); 
	int32_t h, hl = 0; 

	for(int32_t i = 1; i < array.Count(); i++) 
	{
		StringArray ar(array.GetString(i).c_str(), ":"); 
		if(ar.GetString(0) == "@lp")
		{
			if(m_pLearnParams)
				delete m_pLearnParams; 
			m_pLearnParams = new LearnParams(); 
			if(!m_pLearnParams->FromString(ar.GetString(1).c_str()))
				return false; 
		}
		else if(ar.GetString(0) == "@ap")
		{
			if(m_pArchParams)
				delete m_pArchParams; 
			m_pArchParams = new ArchParams_MLP(); 
			if(!m_pArchParams->FromString(ar.GetString(1).c_str()))
				return false; 
			Create(); 
			hl = (int32_t)((ArchParams_MLP*)m_pArchParams)->vtr_hiddens.size(); 	
		}
		else if(ar.GetString(0).find("@w_") == 0)
		{
			StringArray ar_h(ar.GetString(0).c_str(), "_"); 
			sscanf(ar_h.GetString(ar_h.Count() - 1).c_str(), "%d", &h);
			if(h > hl)
				continue; 	
			if(!m_ws[h].FromString(ar.GetString(1).c_str()))
				return false; 
		}
	}

	return true; 
}


string MLP::ConvToModelString()
{
	if(IsNull())
		return string(""); 
	string str("mlp");
	str += "|@lp:"; 
	str += m_pLearnParams->ToString(); 
	str += "|@ap:"; 
	str += m_pArchParams->ToString(); 
	ArchParams_MLP*	p_arch_params = (ArchParams_MLP*)m_pArchParams;
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size();  	
	char stmp[32]; 	
	for(int32_t h = 0; h <= hl; h++)
	{
		sprintf(stmp, "|@w_%d:", h); 
		str += stmp; 
		str += m_ws[h].ToString(); 
	}
	return str; 
}


bool MLP::CompArchWith(NeuralNetwork* pNN)
{
	if(!pNN)
		return false; 
	if(pNN->GetNNType() != m_eNNType)
		return false;
	MLP* p = (MLP*)pNN; 
	if(!p->GetArchParams() || !m_pArchParams)
		return false;
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	ArchParams_MLP* p_arch_params_1 = (ArchParams_MLP*)p->GetArchParams(); 
	if(p_arch_params->input != p_arch_params_1->input || p_arch_params->output != p_arch_params_1->output)
		return false;
	if(p_arch_params->vtr_hiddens.size() != p_arch_params_1->vtr_hiddens.size())
		return false; 
	for(size_t i = 0; i < p_arch_params->vtr_hiddens.size(); i++)
	{
		if(p_arch_params->vtr_hiddens[i] != p_arch_params_1->vtr_hiddens[i])
			return false; 
	}	

	return true; 
}


bool MLP::IsNull()
{
	if(!m_pLearnParams || !m_pArchParams || !m_ws)
		return true; 
	return false; 
} 



NNAssi** MLP::CreateAssi(const bool bForTrain)
{
	if(IsNull())
		return NULL; 
	if(m_pArchParams->GetType() != _ARCH_PARAMS_MLP)
		return NULL; 

	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams;  
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size();
	NNAssi** pp_assi = new NNAssi*[hl+1]; 

	for(int32_t h = 0; h <= hl; h++)
	{
		if(h == 0)
			pp_assi[h] = NNAssi::New(m_pLearnParams->optim, p_arch_params->input, p_arch_params->vtr_hiddens[h], bForTrain); 
		else if(h == hl)
			pp_assi[h] = NNAssi::New(m_pLearnParams->optim, p_arch_params->vtr_hiddens[h-1], p_arch_params->output, bForTrain); 
		else
			pp_assi[h] = NNAssi::New(m_pLearnParams->optim, p_arch_params->vtr_hiddens[h-1], p_arch_params->vtr_hiddens[h], bForTrain); 
	}

	return pp_assi; 
}


bool MLP::ReleaseAssi(NNAssi** ppAssi)
{
	if(IsNull())
		return false; 
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams;
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size();
	if(ppAssi)
	{	
		for(int32_t h = 0; h <= hl; h++) 
		{
			if(ppAssi[h])
				delete ppAssi[h]; 
		}
		delete [] ppAssi; 
	}
	return true; 
}


bool MLP::Create()
{
	Release(); 
	if(!m_pArchParams)
		return false;

	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams;
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size(); 
	m_ws = new Matrix[hl+1]; 

	for(int32_t h = 0; h <= hl; h++) 
	{
		if(h == 0)
			m_ws[h].Create(p_arch_params->input + 1, p_arch_params->vtr_hiddens[h]); // add 1 for bias 
		else if(h == hl)
			m_ws[h].Create(p_arch_params->vtr_hiddens[h-1] + 1, p_arch_params->output); // add 1 for bias 
		else
			m_ws[h].Create(p_arch_params->vtr_hiddens[h-1] + 1, p_arch_params->vtr_hiddens[h]); // add 1 for bias 
		NeuralNetwork::WeightMatrixInit(m_ws[h]); 
	}

	return true; 
} 


void MLP::Release()
{
	if(m_ws)
	{
		delete [] m_ws; 
		m_ws = NULL; 
	}
}


bool MLP::FeedForward(NNAssi** ppAssi, const double* x, const int32_t x_len)
{
	if(IsNull() || !ppAssi || !x)
		return false; 
	if(m_pArchParams->GetType() != _ARCH_PARAMS_MLP)
		return false;

	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	if(x_len != p_arch_params->input)
		return false; 

	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size(); 

	for(int32_t h = 0; h <= hl; h++) 
	{
		if(h == 0)
			LayerActivation(ppAssi[h]->m_ao, x, m_ws[h], p_arch_params->hidden_act);
		else if(h == hl)
			LayerActivation(ppAssi[h]->m_ao, ppAssi[h-1]->m_ao, m_ws[h], OutputActType(p_arch_params->output)); 
		else 
			LayerActivation(ppAssi[h]->m_ao, ppAssi[h-1]->m_ao, m_ws[h], p_arch_params->hidden_act);
	}

	return true; 
}


bool MLP::LayerActivation(double* ao, const double* ai, Matrix& w, const EActType eActType)
{
	if(!ao || !ai || w.IsNull())
		return false; 

	double e = 0.0; 
	for(int32_t j = 0; j < w.Cols(); j++) 
	{
		// forward propagation
		ao[j] = w[w.Rows()-1][j]; 	// bias
		for(int32_t i = 0; i < w.Rows() - 1; i++)
			ao[j] += w[i][j] * ai[i];

		// activation	
		if(eActType == _ACT_SOFTMAX)
			e += exp(ao[j]); 
		else
			ao[j] = Activation::Activate(ao[j], eActType); 
	}	
	if(eActType == _ACT_SOFTMAX)
	{ // softmax
		for(int32_t j = 0; j < w.Cols(); j++) 
			ao[j] = exp(ao[j]) / e; 
	}

	return true; 
}


bool MLP::BackPropagate(double& dLoss, NNAssi** ppAssi, const double* x, const int32_t x_len, const double* y, const int32_t y_len)
{
	if(IsNull() || !ppAssi || !x || !y) 
		return false; 
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	if(x_len != p_arch_params->input || y_len != p_arch_params->output)
		return false; 
	dLoss = 0.0;
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size();  

	// caculate delta of output layer
	for(int32_t j = 0; j < y_len; j++)
		ppAssi[hl]->m_do[j] = ppAssi[hl]->m_ao[j] - y[j]; 	

	int32_t cc = (y_len == 2) ? 1 : y_len; 

	// delta back propagation
	if(cc == 1) 
		LayerDeltaBack(ppAssi[hl-1]->m_do, ppAssi[hl]->m_do, m_ws[hl], true); 
	else
		LayerDeltaBack(ppAssi[hl-1]->m_do, ppAssi[hl]->m_do, m_ws[hl]); 
	for(int32_t h = hl - 2 ; h >= 0; h--)
		LayerDeltaBack(ppAssi[h]->m_do, ppAssi[h+1]->m_do, m_ws[h+1]); 

	// update the gradient matrix of loss function of output layer
	if(m_pArchParams->output == 1)	
	{ // regression, Linear as output activation, quadratic as loss
		ppAssi[hl]->m_g[p_arch_params->vtr_hiddens[hl-1]][0] += ppAssi[hl]->m_do[0] * Activation::DActivate(ppAssi[hl]->m_ao[0], _ACT_LINEAR) * 1.0;
		for(int32_t i = 0; i < p_arch_params->vtr_hiddens[hl-1]; i++)
			ppAssi[hl]->m_g[i][0] += ppAssi[hl]->m_do[0] * Activation::DActivate(ppAssi[hl]->m_ao[0], _ACT_LINEAR) * ppAssi[hl-1]->m_ao[i]; 
		dLoss = Activation::Loss_Quadratic(ppAssi[hl]->m_ao[0], y[0]); // quadratic
	}
	else if(m_pArchParams->output == 2)  
	{ // bi-classification, Sigmoid as output activation, cross entropy as loss
		ppAssi[hl]->m_g[p_arch_params->vtr_hiddens[hl-1]][0] += ppAssi[hl]->m_do[0] * 1.0; 
		for(int32_t i = 0; i < p_arch_params->vtr_hiddens[hl-1]; i++)
			ppAssi[hl]->m_g[i][0] += ppAssi[hl]->m_do[0] * ppAssi[hl-1]->m_ao[i]; 
		dLoss = Activation::Loss_CrossEntropy(ppAssi[hl]->m_ao[0], y[0]); // cross entropy
	}
	else
	{ // multi-classification, Softmax as output activation, log likelihood as loss
		for(int32_t j = 0; j < y_len; j++) 	
		{
			ppAssi[hl]->m_g[p_arch_params->vtr_hiddens[hl-1]][j] += (ppAssi[hl]->m_ao[j] - 1.0 ) * 1.0; 
			for(int32_t i = 0; i < x_len; i++)
				ppAssi[hl]->m_g[i][j] += (ppAssi[hl]->m_ao[j] - 1.0 ) * ppAssi[hl-1]->m_ao[i]; 
		}
		dLoss = Activation::Loss_LogLikelihood(ppAssi[hl]->m_ao, y, y_len);  // log likelihood 
	}

	// update the gradient matrices of loss function of hidden layers
	for(int32_t h = hl - 1; h > 0; h--)
	{
		for(int32_t j = 0; j < p_arch_params->vtr_hiddens[h]; j++) 
		{
			ppAssi[h]->m_g[p_arch_params->vtr_hiddens[h]][j] += ppAssi[h]->m_do[j] * Activation::DActivate(ppAssi[h]->m_ao[j], p_arch_params->hidden_act) * 1.0; 
			for(int32_t i = 0; i < p_arch_params->vtr_hiddens[h-1]; i++) 
				ppAssi[h]->m_g[i][j] += ppAssi[h]->m_do[j] * Activation::DActivate(ppAssi[h]->m_ao[j], p_arch_params->hidden_act) * ppAssi[h-1]->m_ao[i]; 	
		}
	}
	// the lowest hidden layer
	for(int32_t j = 0; j < p_arch_params->vtr_hiddens[0]; j++)
	{
		ppAssi[0]->m_g[p_arch_params->input][j] += ppAssi[0]->m_do[j] * Activation::DActivate(ppAssi[0]->m_ao[j], p_arch_params->hidden_act) * 1.0; 
		for(int32_t i = 0; i < x_len; i++) 
			ppAssi[0]->m_g[i][j] += ppAssi[0]->m_do[j] * Activation::DActivate(ppAssi[0]->m_ao[j], p_arch_params->hidden_act) * x[i]; 	
	}

	return true; 
}


bool MLP::LayerDeltaBack(double* low_do, const double* up_do, Matrix& w, const bool bOneCol) 
{
	if(!low_do || !up_do || w.IsNull())
		return false; 

	for(int32_t i = 0; i < w.Rows() - 1; i++) 
	{ 
		low_do[i] = up_do[0] * w[i][0];
		if(!bOneCol) 
		{
			for(int32_t j = 1; j < w.Cols(); j++) 
				low_do[i] += up_do[j] * w[i][j]; 
		}
	}

	return true; 
}


bool MLP::ModelUpdate(NNAssi** ppAssi, const double dLearningRate) 
{
	if(m_pLearnParams->optim == _OPTIM_SGD)
	{
		SGDParams* p_opt_params = (SGDParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_SGD(ppAssi, dLearningRate, p_opt_params->regula, p_opt_params->lambda); 
	}
	else if(m_pLearnParams->optim == _OPTIM_MOMENTUM)
	{
		MomentumParams* p_opt_params = (MomentumParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_Momentum(ppAssi, dLearningRate, p_opt_params->beta); 
	}
	else if(m_pLearnParams->optim == _OPTIM_NAG)
	{
		NAGParams* p_opt_params = (NAGParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_NAG(ppAssi, dLearningRate, p_opt_params->beta); 
	}
	else if(m_pLearnParams->optim == _OPTIM_ADAGRAD)
	{
		AdaGradParams* p_opt_params = (AdaGradParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_AdaGrad(ppAssi, dLearningRate, p_opt_params->eps); 
	}
	else if(m_pLearnParams->optim == _OPTIM_RMSPROP)
	{
		RMSpropParams* p_opt_params = (RMSpropParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_RMSprop(ppAssi, dLearningRate, p_opt_params->beta, p_opt_params->eps); 
	}
	else if(m_pLearnParams->optim == _OPTIM_ADADELTA)
	{
		AdaDeltaParams* p_opt_params = (AdaDeltaParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_AdaDelta(ppAssi, p_opt_params->rho, p_opt_params->beta, p_opt_params->eps); 
	}
	else if(m_pLearnParams->optim == _OPTIM_ADAM)
	{
		AdamParams* p_opt_params = (AdamParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_Adam(ppAssi, dLearningRate, p_opt_params->beta1, p_opt_params->beta2, p_opt_params->eps); 
	}
	else 
		return false; 
}


bool MLP::ModelUpdate_SGD(NNAssi** ppAssi, const double dLearningRate, const ERegula eRegula, const double dLambda)
{
	if(!ppAssi)
		return false;
	if(ppAssi[0]->m_eOptim != _OPTIM_SGD)
		return false; 
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	NNAssi_SGD* p_assi = NULL; 
	double v;  
	int32_t cc = (p_arch_params->output == 2) ? 1 : p_arch_params->output; 
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size(); 
	int32_t u_size, l_size; 

	// between top hidden and output layer
	p_assi = (NNAssi_SGD*)ppAssi[hl]; 
	for(int32_t i = 0; i < p_arch_params->vtr_hiddens[hl-1] + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			p_assi->m_g[i][j] = p_assi->m_g[i][j] + Activation::DActRegula(m_ws[hl][i][j], eRegula, dLambda); // gradient of regularization
			v = (0.0 - dLearningRate * p_assi->m_g[i][j]);	 	// direction of descent 
			m_ws[hl][i][j] += v; 		// update the weight
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(p_arch_params->output == 2)
			m_ws[hl][i][1] = 0.0 - m_ws[hl][i][0];
	}

	// between hidden layers or between input and bottom hidden layer
	for(int32_t h = hl-1; h >= 0; h--) 
	{
		p_assi = (NNAssi_SGD*)ppAssi[h]; 
		u_size = p_arch_params->vtr_hiddens[h];    
		l_size = (h == 0) ? p_arch_params->input: p_arch_params->vtr_hiddens[h-1];    

		for(int32_t i = 0; i < l_size + 1; i++)	// include bias 
		{
			for(int32_t j = 0; j < u_size; j++) 
			{
				p_assi->m_g[i][j] = p_assi->m_g[i][j] + Activation::DActRegula(m_ws[h][i][j], eRegula, dLambda); // gradient of regularization
				v = (0.0 - dLearningRate * p_assi->m_g[i][j]);	 	// direction of descent 
				m_ws[h][i][j] += v; 		// update the weight
				p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
			}
		}

	}

	return true; 
}


bool MLP::ModelUpdate_Momentum(NNAssi** ppAssi, const double dLearningRate, const double dBeta)
{
	if(!ppAssi)
		return false;
	if(ppAssi[0]->m_eOptim != _OPTIM_MOMENTUM)
		return false; 
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	NNAssi_Momentum* p_assi = NULL; 
	double v;  
	int32_t cc = (p_arch_params->output == 2) ? 1 : p_arch_params->output; 
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size(); 
	int32_t u_size, l_size; 

	// between top hidden and output layer
	p_assi = (NNAssi_Momentum*)ppAssi[hl]; 
	for(int32_t i = 0; i < p_arch_params->vtr_hiddens[hl-1] + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			v = dBeta * p_assi->m_v_prev[i][j] - dLearningRate * p_assi->m_g[i][j]; // direction of descent 	
			m_ws[hl][i][j] += v;			// update the weight
			p_assi->m_v_prev[i][j] = v;	// record last direction 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(p_arch_params->output == 2)
			m_ws[hl][i][1] = 0.0 - m_ws[hl][i][0];
	}

	// between hidden layers or between input and bottom hidden layer
	for(int32_t h = hl-1; h >= 0; h--) 
	{
		p_assi = (NNAssi_Momentum*)ppAssi[h]; 
		u_size = p_arch_params->vtr_hiddens[h];    
		l_size = (h == 0) ? p_arch_params->input: p_arch_params->vtr_hiddens[h-1];    

		for(int32_t i = 0; i < l_size + 1; i++)	// include bias 
		{
			for(int32_t j = 0; j < u_size; j++) 
			{
				v = dBeta * p_assi->m_v_prev[i][j] - dLearningRate * p_assi->m_g[i][j]; // direction of descent 	
				m_ws[h][i][j] += v;			// update the weight
				p_assi->m_v_prev[i][j] = v;	// record last direction 
				p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
			}
		}

	}

	return true; 
}


bool MLP::ModelUpdate_NAG(NNAssi** ppAssi, const double dLearningRate, const double dBeta)
{
	if(!ppAssi)
		return false;
	if(ppAssi[0]->m_eOptim != _OPTIM_NAG)
		return false; 
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	NNAssi_NAG* p_assi = NULL; 
	double v;  
	int32_t cc = (p_arch_params->output == 2) ? 1 : p_arch_params->output; 
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size(); 
	int32_t u_size, l_size; 

	// between top hidden and output layer
	p_assi = (NNAssi_NAG*)ppAssi[hl]; 
	for(int32_t i = 0; i < p_arch_params->vtr_hiddens[hl-1] + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			v = dBeta * p_assi->m_v_prev[i][j] - dLearningRate * p_assi->m_g[i][j];	// direction of descent 
			m_ws[hl][i][j] += (1.0 + dBeta) * v - dBeta * p_assi->m_v_prev[i][j]; 	// update the weight	
			p_assi->m_v_prev[i][j] = v;	// record last direction 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(p_arch_params->output == 2)
			m_ws[hl][i][1] = 0.0 - m_ws[hl][i][0];
	}

	// between hidden layers or between input and bottom hidden layer
	for(int32_t h = hl-1; h >= 0; h--) 
	{
		p_assi = (NNAssi_NAG*)ppAssi[h]; 
		u_size = p_arch_params->vtr_hiddens[h];    
		l_size = (h == 0) ? p_arch_params->input: p_arch_params->vtr_hiddens[h-1];    

		for(int32_t i = 0; i < l_size + 1; i++)	// include bias 
		{
			for(int32_t j = 0; j < u_size; j++) 
			{
				v = dBeta * p_assi->m_v_prev[i][j] - dLearningRate * p_assi->m_g[i][j];	// direction of descent 
				m_ws[h][i][j] += (1.0 + dBeta) * v - dBeta * p_assi->m_v_prev[i][j]; 	// update the weight	
				p_assi->m_v_prev[i][j] = v;	// record last direction 
				p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
			}
		}

	}

	return true; 
}


bool MLP::ModelUpdate_AdaGrad(NNAssi** ppAssi, const double dLearningRate, const double dEps)
{
	if(!ppAssi)
		return false;
	if(ppAssi[0]->m_eOptim != _OPTIM_ADAGRAD)
		return false; 
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	NNAssi_AdaGrad* p_assi = NULL; 
	double v;  
	int32_t cc = (p_arch_params->output == 2) ? 1 : p_arch_params->output; 
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size(); 
	int32_t u_size, l_size; 

	// between top hidden and output layer
	p_assi = (NNAssi_AdaGrad*)ppAssi[hl]; 
	for(int32_t i = 0; i < p_arch_params->vtr_hiddens[hl-1] + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			p_assi->m_g2_acc[i][j] += p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
			v = 0.0 - dLearningRate * p_assi->m_g[i][j] / (sqrt(p_assi->m_g2_acc[i][j]) + dEps);
			m_ws[hl][i][j] += v; 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(p_arch_params->output == 2)
			m_ws[hl][i][1] = 0.0 - m_ws[hl][i][0];
	}

	// between hidden layers or between input and bottom hidden layer
	for(int32_t h = hl-1; h >= 0; h--) 
	{
		p_assi = (NNAssi_AdaGrad*)ppAssi[h]; 
		u_size = p_arch_params->vtr_hiddens[h];    
		l_size = (h == 0) ? p_arch_params->input: p_arch_params->vtr_hiddens[h-1];    

		for(int32_t i = 0; i < l_size + 1; i++)	// include bias 
		{
			for(int32_t j = 0; j < u_size; j++) 
			{
				p_assi->m_g2_acc[i][j] += p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
				v = 0.0 - dLearningRate * p_assi->m_g[i][j] / (sqrt(p_assi->m_g2_acc[i][j]) + dEps);
				m_ws[h][i][j] += v; 
				p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
			}
		}
	}

	return true; 
}


bool MLP::ModelUpdate_RMSprop(NNAssi** ppAssi, const double dLearningRate, const double dBeta, const double dEps)
{
	if(!ppAssi)
		return false;
	if(ppAssi[0]->m_eOptim != _OPTIM_RMSPROP)
		return false; 
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	NNAssi_RMSprop* p_assi = NULL; 
	double v;  
	int32_t cc = (p_arch_params->output == 2) ? 1 : p_arch_params->output; 
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size(); 
	int32_t u_size, l_size; 

	// between top hidden and output layer
	p_assi = (NNAssi_RMSprop*)ppAssi[hl]; 
	for(int32_t i = 0; i < p_arch_params->vtr_hiddens[hl-1] + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			p_assi->m_g2_mavg[i][j] = dBeta * p_assi->m_g2_mavg[i][j] + (1.0 - dBeta) * p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
			v = 0.0 - dLearningRate * p_assi->m_g[i][j] / (sqrt(p_assi->m_g2_mavg[i][j]) + dEps);
			m_ws[hl][i][j] += v; 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(p_arch_params->output == 2)
			m_ws[hl][i][1] = 0.0 - m_ws[hl][i][0];
	}

	// between hidden layers or between input and bottom hidden layer
	for(int32_t h = hl-1; h >= 0; h--) 
	{
		p_assi = (NNAssi_RMSprop*)ppAssi[h]; 
		u_size = p_arch_params->vtr_hiddens[h];    
		l_size = (h == 0) ? p_arch_params->input: p_arch_params->vtr_hiddens[h-1];    

		for(int32_t i = 0; i < l_size + 1; i++)	// include bias 
		{
			for(int32_t j = 0; j < u_size; j++) 
			{
				p_assi->m_g2_mavg[i][j] = dBeta * p_assi->m_g2_mavg[i][j] + (1.0 - dBeta) * p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
				v = 0.0 - dLearningRate * p_assi->m_g[i][j] / (sqrt(p_assi->m_g2_mavg[i][j]) + dEps);
				m_ws[h][i][j] += v; 
				p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
			}
		}
	}

	return true; 
}


bool MLP::ModelUpdate_AdaDelta(NNAssi** ppAssi, const double dRho, const double dBeta, const double dEps)
{
	if(!ppAssi)
		return false;
	if(ppAssi[0]->m_eOptim != _OPTIM_ADADELTA)
		return false; 
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	NNAssi_AdaDelta* p_assi = NULL; 
	double v;  
	int32_t cc = (p_arch_params->output == 2) ? 1 : p_arch_params->output; 
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size(); 
	int32_t u_size, l_size; 

	// between top hidden and output layer
	p_assi = (NNAssi_AdaDelta*)ppAssi[hl]; 
	for(int32_t i = 0; i < p_arch_params->vtr_hiddens[hl-1] + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			p_assi->m_g2_mavg[i][j] = dBeta * p_assi->m_g2_mavg[i][j] + (1.0 - dBeta) * p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
			v = 0.0 - sqrt(p_assi->m_v2_mavg[i][j] + dEps) / sqrt(p_assi->m_g2_mavg[i][j] + dEps) * p_assi->m_g[i][j];  	
			m_ws[hl][i][j] += v; 
			p_assi->m_v2_mavg[i][j] = dRho * p_assi->m_v2_mavg[i][j] + (1.0 - dRho) * v * v; 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(p_arch_params->output == 2)
			m_ws[hl][i][1] = 0.0 - m_ws[hl][i][0];
	}

	// between hidden layers or between input and bottom hidden layer
	for(int32_t h = hl-1; h >= 0; h--) 
	{
		p_assi = (NNAssi_AdaDelta*)ppAssi[h]; 
		u_size = p_arch_params->vtr_hiddens[h];    
		l_size = (h == 0) ? p_arch_params->input: p_arch_params->vtr_hiddens[h-1];    

		for(int32_t i = 0; i < l_size + 1; i++)	// include bias 
		{
			for(int32_t j = 0; j < u_size; j++) 
			{
				p_assi->m_g2_mavg[i][j] = dBeta * p_assi->m_g2_mavg[i][j] + (1.0 - dBeta) * p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
				v = 0.0 - sqrt(p_assi->m_v2_mavg[i][j] + dEps) / sqrt(p_assi->m_g2_mavg[i][j] + dEps) * p_assi->m_g[i][j];  	
				m_ws[h][i][j] += v; 
				p_assi->m_v2_mavg[i][j] = dRho * p_assi->m_v2_mavg[i][j] + (1.0 - dRho) * v * v; 
				p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
			}
		}
	}

	return true; 
}


bool MLP::ModelUpdate_Adam(NNAssi** ppAssi, const double dLearningRate, const double dBeta1, const double dBeta2, const double dEps)
{
	if(!ppAssi)
		return false;
	if(ppAssi[0]->m_eOptim != _OPTIM_ADAM)
		return false; 
	ArchParams_MLP* p_arch_params = (ArchParams_MLP*)m_pArchParams; 
	NNAssi_Adam* p_assi = NULL; 
	double v;  
	int32_t cc = (p_arch_params->output == 2) ? 1 : p_arch_params->output; 
	int32_t hl = (int32_t)p_arch_params->vtr_hiddens.size(); 
	int32_t u_size, l_size; 

	// between top hidden and output layer
	p_assi = (NNAssi_Adam*)ppAssi[hl]; 
	for(int32_t i = 0; i < p_arch_params->vtr_hiddens[hl-1] + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			p_assi->m_g_mavg[i][j] = dBeta1 * p_assi->m_g_mavg[i][j] + (1.0 - dBeta1) * p_assi->m_g[i][j]; 
			p_assi->m_g2_mavg[i][j] = dBeta2 * p_assi->m_g2_mavg[i][j] + (1.0 - dBeta2) * p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
			v = 0.0 - dLearningRate * p_assi->m_g_mavg[i][j] / (sqrt(p_assi->m_g2_mavg[i][j]) + dEps);
			m_ws[hl][i][j] += v; 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(p_arch_params->output == 2)
			m_ws[hl][i][1] = 0.0 - m_ws[hl][i][0];
	}

	// between hidden layers or between input and bottom hidden layer
	for(int32_t h = hl-1; h >= 0; h--) 
	{
		p_assi = (NNAssi_Adam*)ppAssi[h]; 
		u_size = p_arch_params->vtr_hiddens[h];    
		l_size = (h == 0) ? p_arch_params->input: p_arch_params->vtr_hiddens[h-1];    

		for(int32_t i = 0; i < l_size + 1; i++)	// include bias 
		{
			for(int32_t j = 0; j < u_size; j++) 
			{
				p_assi->m_g_mavg[i][j] = dBeta1 * p_assi->m_g_mavg[i][j] + (1.0 - dBeta1) * p_assi->m_g[i][j]; 
				p_assi->m_g2_mavg[i][j] = dBeta2 * p_assi->m_g2_mavg[i][j] + (1.0 - dBeta2) * p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
				v = 0.0 - dLearningRate * p_assi->m_g_mavg[i][j] / (sqrt(p_assi->m_g2_mavg[i][j]) + dEps);
				m_ws[h][i][j] += v; 
				p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
			}
		}
	}

	return true; 
}




