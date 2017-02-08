#include "Perceptron.h"
#include "Activation.h"
using namespace metis_nn; 
#include "Config.h"
#include "StringArray.h"
using namespace metis_uti; 
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std; 
#include <string.h>
#include <math.h>


Perceptron::Perceptron() : NeuralNetwork(_NN_PERCEPTRON)
{
}


Perceptron::~Perceptron()
{
	Release(); 
}


bool Perceptron::Init(LearnParams* pLearnParams, ArchParams* pArchParams)
{
	if(!pLearnParams || !pArchParams)
		return false; 
	if(pArchParams->GetType() != _ARCH_PARAMS_BASIC)
		return false; 

	if(m_pLearnParams)
		delete m_pLearnParams; 
	m_pLearnParams = new LearnParams(*pLearnParams); 
	if(m_pArchParams)
		delete m_pArchParams; 
	m_pArchParams = new ArchParams(*pArchParams); 
	Create(); 

	return true; 
}


bool Perceptron::InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	if(conf.ValCnt("Model") == 0)
		return false;
	if(conf.GetVal_asString("Model") != "Perceptron")
		return false;
	if(m_pLearnParams)	
		delete m_pLearnParams; 
	m_pLearnParams = new LearnParams(); 
	if(m_pArchParams)
		delete m_pArchParams; 
	m_pArchParams = new ArchParams(); 

	if(!m_pArchParams->FromConfig(sConfigFile, nInput, nOutput))
		return false; 
	if(!m_pLearnParams->FromConfig(sConfigFile))
		return false; 

	Create(); 
	return true; 
}


bool Perceptron::Epoch(double& dAvgLoss, NNAssi** ppAssi, vector<Pattern*>& vtrPatts, const double dLearningRate, const int32_t nStartOff) 
{
	if(IsNull() || !ppAssi)
		return false;
	NNAssi* p_assi = ppAssi[0]; 
	if(p_assi == NULL)
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
		FeedForward(p_assi, vtrPatts[t]->m_x, vtrPatts[t]->m_nXCnt); 
		BackPropagate(loss, p_assi, vtrPatts[t]->m_x, vtrPatts[t]->m_nXCnt, vtrPatts[t]->m_y, vtrPatts[t]->m_nYCnt); 
		dAvgLoss += loss; 
		patt_cnt++; 

		if(patt_cnt >= batch_cnt)
		{
			ModelUpdate(p_assi, dLearningRate); 
			patt_cnt = 0; 	
		}
	}
	if(patt_cnt > 0)
	{
		ModelUpdate(p_assi, dLearningRate); 
		patt_cnt = 0; 	
	}

	dAvgLoss /= (double)((int32_t)vtrPatts.size() - nStartOff); 
	return true; 
}


int32_t Perceptron::Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len)
{
	if(!y || !x)
		return _METIS_NN_ERROR_INPUT_NULL; 
	if(IsNull())
		return _METIS_NN_ERROR_MODEL_NULL; 
	if(x_len < 0 || x_len != m_pArchParams->input || y_len < 0)
		return _METIS_NN_ERROR_WRONG_LEN; 
	NNAssi* p_assi = NNAssi::New(m_pLearnParams->optim, m_pArchParams->input, m_pArchParams->output, false); 
	if(!p_assi)
		return _METIS_NN_ASSI_ERROR; 		

	FeedForward(p_assi, x, x_len); 

	for(int32_t j = 0; j < y_len; j++)
	{
		if(j < m_pArchParams->output)
			y[j] = p_assi->m_ao[j];
		else
			y[j] = 0.0; 
	}

	delete p_assi;
	return _METIS_NN_SUCCESS; 
}


int32_t Perceptron::Save(const char* sModelFile)
{
	if(!sModelFile)
		return _METIS_NN_ERROR_INPUT_NULL; 
	if(IsNull())
		return _METIS_NN_ERROR_MODEL_NULL; 
	ofstream ofs(sModelFile); 
	if(!ofs.is_open())
		return _METIS_NN_ERROR_FILE_OPEN;

	ofs<<"** Perceptron **"<<endl; 
	ofs<<endl;

	// save learning parameters
	ofs<<"@learning_params"<<endl; 
	m_pLearnParams->Print(ofs); 
	ofs<<endl; 

	// save architecture parameters of perceptron
	ofs<<"@architecture_params"<<endl; 
	m_pArchParams->Print(ofs); 
	ofs<<endl; 

	// save transtorm matrix
	ofs<<"@weight"<<endl; 
	Matrix::Print_Matrix(ofs, m_w);
	ofs<<endl; 

	ofs.close(); 
	return _METIS_NN_SUCCESS; 
}


int32_t Perceptron::Load(const char* sModelFile)
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
	m_pArchParams = new ArchParams(); 
	Release(); 	

	string str; 

	std::getline(ifs, str); 
	if(str != "** Perceptron **")
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
		}
		else if(str == "@weight")
		{
			if(!Matrix::Read_Matrix(m_w, ifs))
				return _METIS_NN_ERROR_MODEL_DATA;
		}
	}

	ifs.close(); 
	return _METIS_NN_SUCCESS; 
}


bool Perceptron::SetByModelString(const char* sModelStr)
{
	if(!sModelStr)
		return false;

	StringArray array(sModelStr, "|"); 
	if(array.GetString(0) != "perceptron")	
		return false; 	
	if(array.Count() < 4)
		return false; 

	Release(); 

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
			m_pArchParams = new ArchParams(); 
			if(!m_pArchParams->FromString(ar.GetString(1).c_str()))
				return false; 
		}
		else if(ar.GetString(0) == "@w")
		{
			if(!m_w.FromString(ar.GetString(1).c_str()))
				return false; 
		}
	}

	return true; 
}


string Perceptron::ConvToModelString()
{
	if(IsNull())
		return string(""); 
	string str("perceptron");
	str += "|@lp:"; 
	str += m_pLearnParams->ToString(); 
	str += "|@ap:"; 
	str += m_pArchParams->ToString(); 
	str += "|@w:"; 
	str += m_w.ToString(); 
	return str; 
}


bool Perceptron::CombineWith(NeuralNetwork* pNN, const double w0, const double w1)
{
	if(!CompArchWith(pNN))
		return false; 
	Perceptron* p = (Perceptron*)pNN; 
	return m_w.CombineWith(p->m_w, w0, w1); 
}


void Perceptron::NumericMultiWith(const double a)
{
	m_w.NumericMultiWith(a); 
}


bool Perceptron::CompArchWith(NeuralNetwork* pNN)
{
	if(!pNN)
		return false; 
	if(pNN->GetNNType() != m_eNNType)
		return false;
	Perceptron* p = (Perceptron*)pNN; 
	if(!p->GetArchParams() || !m_pArchParams)
		return false; 
	if(p->GetArchParams()->input != m_pArchParams->input || p->GetArchParams()->output != m_pArchParams->output)
		return false; 
	return true; 
}


bool Perceptron::IsNull()
{
	if(!m_pLearnParams || !m_pArchParams || m_w.IsNull())
		return true; 
	return false; 
}

	
NNAssi** Perceptron::CreateAssi(const bool bForTrain)
{
	if(IsNull())
		return NULL; 
	if(m_pArchParams->GetType() != _ARCH_PARAMS_BASIC)
		return NULL; 
	NNAssi** pp_assi = new NNAssi*[1];
	pp_assi[0] = NNAssi::New(m_pLearnParams->optim, m_pArchParams->input, m_pArchParams->output, bForTrain); 
	return pp_assi;  
}


bool Perceptron::ReleaseAssi(NNAssi** ppAssi)
{
	if(IsNull())
		return false; 
	if(ppAssi)
	{
		if(ppAssi[0])
			delete ppAssi[0];
		delete [] ppAssi; 
	}	
	return true; 
}


bool Perceptron::Create()
{
	Release(); 
	if(!m_pArchParams)
		return false; 
	m_w.Create(m_pArchParams->input + 1, m_pArchParams->output);	// add 1 for bias 
	NeuralNetwork::WeightMatrixInit(m_w); 
	return true; 
}


void Perceptron::Release()
{
	m_w.Release(); 
}


bool Perceptron::FeedForward(NNAssi* pAssi, const double* x, const int32_t x_len)
{
	if(IsNull() || !pAssi || !x || x_len != m_pArchParams->input)
		return false; 

	EActType e_out_act = NeuralNetwork::OutputActType(m_pArchParams->output); 
	double e = 0.0;

	for(int32_t j = 0; j < m_pArchParams->output; j++) 
	{
		// forward propagation
		pAssi->m_ao[j] = m_w[m_pArchParams->input][j];	// bias
		for(int32_t i = 0; i < x_len; i++) 
			pAssi->m_ao[j] += m_w[i][j] * x[i]; 

		// activation	
		if(e_out_act == _ACT_SOFTMAX)
			e += exp(pAssi->m_ao[j]); 
		else 
			pAssi->m_ao[j] = Activation::Activate(pAssi->m_ao[j], e_out_act); 
	}
	if(e_out_act == _ACT_SOFTMAX)
	{ // softmax
		for(int32_t j = 0; j < m_pArchParams->output; j++)
			pAssi->m_ao[j] = exp(pAssi->m_ao[j]) / e; 
	}

	return true; 
}


bool Perceptron::BackPropagate(double& dLoss, NNAssi* pAssi, const double* x, const int32_t x_len, const double* y, const int32_t y_len) 
{
	if(IsNull() || !pAssi || !x || x_len != m_pArchParams->input || !y || y_len != m_pArchParams->output)
		return false; 

	// calculate delta of output
	for(int32_t j = 0; j < y_len; j++)
		pAssi->m_do[j] = pAssi->m_ao[j] - y[j];

	// update the gradient matrix of loss function
	if(m_pArchParams->output == 1)	
	{ // regression, Linear as output activation, quadratic as loss
		pAssi->m_g[m_pArchParams->input][0] += pAssi->m_do[0] * Activation::DActivate(pAssi->m_ao[0], _ACT_LINEAR) * 1.0;
		for(int32_t i = 0; i < x_len; i++)
			pAssi->m_g[i][0] += pAssi->m_do[0] * Activation::DActivate(pAssi->m_ao[0], _ACT_LINEAR) * x[i]; 
		dLoss = Activation::Loss_Quadratic(pAssi->m_ao[0], y[0]); // quadratic
	}
	else if(m_pArchParams->output == 2)  
	{ // bi-classification, Sigmoid as output activation, cross entropy as loss
		pAssi->m_g[m_pArchParams->input][0] += pAssi->m_do[0] * 1.0; 
		for(int32_t i = 0; i < x_len; i++)
			pAssi->m_g[i][0] += pAssi->m_do[0] * x[i]; 
		dLoss = Activation::Loss_CrossEntropy(pAssi->m_ao[0], y[0]); // cross entropy
	}
	else
	{ // multi-classification, Softmax as output activation, log likelihood as loss
		for(int32_t j = 0; j < y_len; j++) 	
		{
			pAssi->m_g[m_pArchParams->input][j] += (pAssi->m_ao[j] - 1.0 ) * 1.0; 
			for(int32_t i = 0; i < x_len; i++)
				pAssi->m_g[i][j] += (pAssi->m_ao[j] - 1.0 ) * x[i]; 
		}
		dLoss = Activation::Loss_LogLikelihood(pAssi->m_ao, y, y_len);  // log likelihood 
	}

	return true; 
}


bool Perceptron::ModelUpdate(NNAssi* pAssi, const double dLearningRate)
{
	if(m_pLearnParams->optim == _OPTIM_SGD)
	{
		SGDParams* p_opt_params = (SGDParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_SGD(pAssi, dLearningRate, p_opt_params->regula, p_opt_params->lambda); 
	}
	else if(m_pLearnParams->optim == _OPTIM_MOMENTUM)
	{
		MomentumParams* p_opt_params = (MomentumParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_Momentum(pAssi, dLearningRate, p_opt_params->beta); 
	}
	else if(m_pLearnParams->optim == _OPTIM_NAG)
	{
		NAGParams* p_opt_params = (NAGParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_NAG(pAssi, dLearningRate, p_opt_params->beta); 
	}
	else if(m_pLearnParams->optim == _OPTIM_ADAGRAD)
	{
		AdaGradParams* p_opt_params = (AdaGradParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_AdaGrad(pAssi, dLearningRate, p_opt_params->eps);
	}
	else if(m_pLearnParams->optim == _OPTIM_RMSPROP)
	{
		RMSpropParams* p_opt_params = (RMSpropParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_RMSprop(pAssi, dLearningRate, p_opt_params->beta, p_opt_params->eps);
	}
	else if(m_pLearnParams->optim == _OPTIM_ADADELTA)
	{
		AdaDeltaParams* p_opt_params = (AdaDeltaParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_AdaDelta(pAssi, p_opt_params->rho, p_opt_params->beta, p_opt_params->eps); 
	}
	else if(m_pLearnParams->optim == _OPTIM_ADAM)
	{
		AdamParams* p_opt_params = (AdamParams*)m_pLearnParams->p_optim_params; 
		return ModelUpdate_Adam(pAssi, dLearningRate, p_opt_params->beta1, p_opt_params->beta2, p_opt_params->eps); 
	}
	else 
		return false; 
}


bool Perceptron::ModelUpdate_SGD(NNAssi* pAssi, const double dLearningRate, const ERegula eRegula, const double dLambda)
{
	if(pAssi->m_eOptim != _OPTIM_SGD)
		return false;
	NNAssi_SGD* p_assi = (NNAssi_SGD*)pAssi; 
	double v;  
	int32_t cc = (m_pArchParams->output == 2) ? 1 : m_pArchParams->output; 
	for(int32_t i = 0; i < m_pArchParams->input + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			p_assi->m_g[i][j] = p_assi->m_g[i][j] + Activation::DActRegula(m_w[i][j], eRegula, dLambda); // gradient of regularization
			v = (0.0 - dLearningRate * p_assi->m_g[i][j]);	 	// direction of descent 
			m_w[i][j] += v; 		// update the weight
			p_assi->m_g[i][j] = 0.0;		// reset the value of the gradient 
		}
		if(m_pArchParams->output == 2)
			m_w[i][1] = 0.0 - m_w[i][0];
	}
	return true; 
}


bool Perceptron::ModelUpdate_Momentum(NNAssi* pAssi, const double dLearningRate, const double dBeta) 
{
	if(pAssi->m_eOptim != _OPTIM_MOMENTUM)
		return false; 
	NNAssi_Momentum* p_assi = (NNAssi_Momentum*)pAssi; 
	double v;
	int32_t cc = (m_pArchParams->output == 2) ? 1 : m_pArchParams->output; 
	for(int32_t i = 0; i < m_pArchParams->input + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			v = dBeta * p_assi->m_v_prev[i][j] - dLearningRate * p_assi->m_g[i][j]; // direction of descent 	
			m_w[i][j] += v;			// update the weight
			p_assi->m_v_prev[i][j] = v;	// record last direction 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(m_pArchParams->output == 2)
			m_w[i][1] = 0.0 - m_w[i][0];
	}
	return true; 
}


bool Perceptron::ModelUpdate_NAG(NNAssi* pAssi, const double dLearningRate, const double dBeta) 
{
	if(pAssi->m_eOptim != _OPTIM_NAG)
		return false; 
	NNAssi_NAG* p_assi = (NNAssi_NAG*)pAssi; 	
	double v; 
	int32_t cc = (m_pArchParams->output == 2) ? 1 : m_pArchParams->output; 
	for(int32_t i = 0; i < m_pArchParams->input + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			v = dBeta * p_assi->m_v_prev[i][j] - dLearningRate * p_assi->m_g[i][j];	// direction of descent 
			m_w[i][j] += (1.0 + dBeta) * v - dBeta * p_assi->m_v_prev[i][j]; 	// update the weight	
			p_assi->m_v_prev[i][j] = v;	// record last direction 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(m_pArchParams->output == 2)
			m_w[i][1] = 0.0 - m_w[i][0];
	}
	return true; 
}


bool Perceptron::ModelUpdate_AdaGrad(NNAssi* pAssi, const double dLearningRate, const double dEps)
{
	if(pAssi->m_eOptim != _OPTIM_ADAGRAD)
		return false; 
	NNAssi_AdaGrad* p_assi = (NNAssi_AdaGrad*)pAssi; 	
	double v; 
	int32_t cc = (m_pArchParams->output == 2) ? 1 : m_pArchParams->output; 
	for(int32_t i = 0; i < m_pArchParams->input + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			p_assi->m_g2_acc[i][j] += p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
			v = 0.0 - dLearningRate * p_assi->m_g[i][j] / (sqrt(p_assi->m_g2_acc[i][j]) + dEps);
			m_w[i][j] += v; 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(m_pArchParams->output == 2)
			m_w[i][1] = 0.0 - m_w[i][0];
	}
	return true; 
}


bool Perceptron::ModelUpdate_RMSprop(NNAssi* pAssi, const double dLearningRate, const double dBeta, const double dEps)
{
	if(pAssi->m_eOptim != _OPTIM_RMSPROP)
		return false; 
	NNAssi_RMSprop* p_assi = (NNAssi_RMSprop*)pAssi; 	
	double v; 
	int32_t cc = (m_pArchParams->output == 2) ? 1 : m_pArchParams->output; 
	for(int32_t i = 0; i < m_pArchParams->input + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			p_assi->m_g2_mavg[i][j] = dBeta * p_assi->m_g2_mavg[i][j] + (1.0 - dBeta) * p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
			v = 0.0 - dLearningRate * p_assi->m_g[i][j] / (sqrt(p_assi->m_g2_mavg[i][j]) + dEps);
			m_w[i][j] += v; 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(m_pArchParams->output == 2)
			m_w[i][1] = 0.0 - m_w[i][0];
	}
	return true; 
}


bool Perceptron::ModelUpdate_AdaDelta(NNAssi* pAssi, const double dRho, const double dBeta, const double dEps)
{
	if(pAssi->m_eOptim != _OPTIM_ADADELTA)
		return false; 
	NNAssi_AdaDelta* p_assi = (NNAssi_AdaDelta*)pAssi; 	
	double v; 
	int32_t cc = (m_pArchParams->output == 2) ? 1 : m_pArchParams->output; 
	for(int32_t i = 0; i < m_pArchParams->input + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			p_assi->m_g2_mavg[i][j] = dBeta * p_assi->m_g2_mavg[i][j] + (1.0 - dBeta) * p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
			v = 0.0 - sqrt(p_assi->m_v2_mavg[i][j] + dEps) / sqrt(p_assi->m_g2_mavg[i][j] + dEps) * p_assi->m_g[i][j];  	
			m_w[i][j] += v; 
			p_assi->m_v2_mavg[i][j] = dRho * p_assi->m_v2_mavg[i][j] + (1.0 - dRho) * v * v; 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(m_pArchParams->output == 2)
			m_w[i][1] = 0.0 - m_w[i][0];
	}
	return true; 
}


bool Perceptron::ModelUpdate_Adam(NNAssi* pAssi, const double dLearningRate, const double dBeta1, const double dBeta2, const double dEps)
{
	if(pAssi->m_eOptim != _OPTIM_ADAM)
		return false; 
	NNAssi_Adam* p_assi = (NNAssi_Adam*)pAssi; 	
	double v; 
	int32_t cc = (m_pArchParams->output == 2) ? 1 : m_pArchParams->output; 
	for(int32_t i = 0; i < m_pArchParams->input + 1; i++)	// include bias 
	{
		for(int32_t j = 0; j < cc; j++) 
		{
			p_assi->m_g_mavg[i][j] = dBeta1 * p_assi->m_g_mavg[i][j] + (1.0 - dBeta1) * p_assi->m_g[i][j]; 
			p_assi->m_g2_mavg[i][j] = dBeta2 * p_assi->m_g2_mavg[i][j] + (1.0 - dBeta2) * p_assi->m_g[i][j] * p_assi->m_g[i][j]; 
			v = 0.0 - dLearningRate * p_assi->m_g_mavg[i][j] / (sqrt(p_assi->m_g2_mavg[i][j]) + dEps);
			m_w[i][j] += v; 
			p_assi->m_g[i][j] = 0.0;	// reset the value of the gradient 
		}
		if(m_pArchParams->output == 2)
			m_w[i][1] = 0.0 - m_w[i][0];
	}
	return true; 
}





