#include "FMSNN.h"
using namespace metis_nn; 
#include "Timer.h"
#include "StringArray.h"
#include "Config.h"
using namespace metis_uti; 
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std; 
#include <math.h>
#include <stdio.h>


////////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction  

FMSNN::FMSNN()
{
	m_vfs = NULL; 
	m_whs = NULL; 

	m_ai = NULL; 
	m_af = NULL; 
	m_ahs = NULL; 
	m_ao = NULL;

	m_df = NULL; 
	m_dhs = NULL;
	m_do = NULL;

	m_cvfs = NULL; 
	m_chs = NULL;	
}


FMSNN::~FMSNN()
{
	Release(); 
}


////////////////////////////////////////////////////////////////////////////////////////////
// Operations

void FMSNN::Init(const FMSNNParamsT fmsnnParamsT, const FMSNNLearningParamsT fmsnnLearningParamsT)
{
	Release(); 

	m_paramsFMSNN = fmsnnParamsT; 
	m_paramsFMSNN.input += 1;		// add 1 for bias nodes 
	m_paramsLearning = fmsnnLearningParamsT; 

	Create(); 
}


bool FMSNN::InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	Release(); 
	m_paramsFMSNN.input = nInput + 1;		// add 1 for bias nodes
	m_paramsFMSNN.output = nOutput;
	m_paramsFMSNN.vtr_hidden.clear(); 
	//m_paramsFMSNN.fm_layer = conf.GetVal_asInt("FMLayer"); 
	m_paramsFMSNN.fm_inter = conf.GetVal_asInt("FMInter"); 
	m_paramsFMSNN.fm_k = conf.GetVal_asInt("FM_K"); 
	for(int32_t i = 0; i < conf.ValCnt("Hiddens"); i++) 
		m_paramsFMSNN.vtr_hidden.push_back(conf.GetVal_asInt("Hiddens", i)); 
	if(m_paramsFMSNN.vtr_hidden.empty())
		return false; 
	m_paramsFMSNN.act_fm_layer = TypeDefs::ActType(conf.GetVal_asString("ActFMLayer").c_str());   
	m_paramsFMSNN.act_hidden = TypeDefs::ActType(conf.GetVal_asString("ActHidden").c_str());   
	m_paramsFMSNN.act_output = TypeDefs::ActType(conf.GetVal_asString("ActOutput").c_str());   
	if(m_paramsFMSNN.act_hidden == _ACT_NONE || m_paramsFMSNN.act_output == _ACT_NONE)
		return false; 

	m_paramsLearning.regula = TypeDefs::RegulaType(conf.GetVal_asString("Regula").c_str()); 
	m_paramsLearning.mini_batch = conf.GetVal_asInt("MiniBatch"); 
	m_paramsLearning.iterations = conf.GetVal_asInt("Iterations"); 
	m_paramsLearning.learning_rate = conf.GetVal_asFloat("LearningRate"); 
	m_paramsLearning.rate_decay = conf.GetVal_asFloat("RateDecay"); 
	m_paramsLearning.epsilon = conf.GetVal_asFloat("Epsilon"); 

	m_paramsLearning.batch_norm = conf.GetVal_asBool("BatchNorm", 0, false); 

	Create(); 
	return true; 
}


void FMSNN::Train(vector<Pattern*>& vtrPatts)
{
	int32_t train_cnt = (int32_t)vtrPatts.size(); 
	double learning_rate = m_paramsLearning.learning_rate;	// learning rate, it would be update after every iteration
	double last_error = -1.0, error, rmse;	// training error and RMSE in one iteration
	Timer timer;		// timer
	int32_t patt_cnt = 0; 

	// create assistant variables for training
	CreateAssistant();

	for(int32_t t = 0; t < m_paramsLearning.iterations; t++) 
	{
		error = 0.0; 	

		timer.Start(); 	

		// shuffle training patterns
		random_shuffle(vtrPatts.begin(), vtrPatts.end()); 

		for(int32_t p = 0; p < train_cnt; p++) 
		{
			// forward & backward phase
			FeedForward(vtrPatts[p]->m_x, vtrPatts[p]->m_nXCnt); 
			error += BackPropagate(vtrPatts[p]->m_y, vtrPatts[p]->m_nYCnt); 
			patt_cnt++; 

			if(m_paramsLearning.mini_batch > 0)	// online or mini-batch
			{
				if(patt_cnt >= m_paramsLearning.mini_batch)
				{
					ModelUpdate(learning_rate, m_paramsLearning.regula, patt_cnt); 
					patt_cnt = 0; 
				}
			}
		}	

		if(m_paramsLearning.mini_batch == 0)	// batch 
		{
			ModelUpdate(learning_rate, m_paramsLearning.regula, patt_cnt); 
			patt_cnt = 0; 
		}
		rmse = sqrt(error / (double)(train_cnt));
		
		timer.Stop(); 	

		printf("iter %d | learning_rate: %.6g | error: %.6g | rmse: %.6g | time_cost(s): %.3f\n", 
				t+1, learning_rate, error, rmse, timer.GetLast_asSec()); 
		learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * m_paramsLearning.rate_decay)));	
		//if(last_error > 0.0 && error >= last_error)
		//	learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * m_paramsLearning.rate_decay)));	
		last_error = error; 

		if(rmse <= m_paramsLearning.epsilon)
			break;
	}
	printf("** training done! time cost %.3f sec\n", timer.GetTotal_asSec());
}


void FMSNN::TrainAndValidate(vector<Pattern*>& vtrPatts)
{
	int32_t cross_cnt = (int32_t)vtrPatts.size() / 20;			// 5% patterns for corss validation
	int32_t train_cnt = (int32_t)vtrPatts.size() - cross_cnt;	// 95% patterns for training
	double learning_rate = m_paramsLearning.learning_rate;	// learning rate, it would be update after every iteration
	double error, rmse, last_error = -1.0, last_rmse = -1.0;	// training error and RMSE in one iteration
	pair<double,double> validation;	// precision and RMSE of validation 
	Timer timer;		// timer
	int32_t patt_cnt = 0; 
	// create assistant variables for training
	CreateAssistant();

	// shuffle pattens 
	random_shuffle(vtrPatts.begin(), vtrPatts.end());

	for(int32_t t = 0; t < m_paramsLearning.iterations; t++) 
	{
		error = 0.0; 	

		timer.Start(); 	

		// shuffle training patterns
		random_shuffle(vtrPatts.begin(), vtrPatts.end() - cross_cnt);

		for(int32_t p = 0; p < train_cnt; p++) 
		{
			// forward & backward phase
			FeedForward(vtrPatts[p]->m_x, vtrPatts[p]->m_nXCnt); 
			error += BackPropagate(vtrPatts[p]->m_y, vtrPatts[p]->m_nYCnt); 
			patt_cnt++; 

			if(m_paramsLearning.mini_batch > 0)	// online or mini-batch
			{
				if(patt_cnt >= m_paramsLearning.mini_batch)
				{
					ModelUpdate(learning_rate, m_paramsLearning.regula, patt_cnt); 
					patt_cnt = 0; 
				}
			}
		}	

		if(m_paramsLearning.mini_batch == 0)	// batch
		{
			ModelUpdate(learning_rate, m_paramsLearning.regula, patt_cnt); 
			patt_cnt = 0; 
		}

		validation = Validation(vtrPatts, cross_cnt); 
		rmse = sqrt(error / (double)(train_cnt));
		
		timer.Stop(); 	

		printf("iter %d | learning_rate: %.6g | error: %.6g | rmse: %.6g | validation(pr & rmse): %.4g%% & %.6g | time_cost(s): %.3f\n", 
				t+1, learning_rate, error, rmse, validation.first * 100.0, validation.second, timer.GetLast_asSec()); 
		learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * m_paramsLearning.rate_decay)));	
		//if(last_error > 0.0 && error >= last_error)
		//	learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * m_paramsLearning.rate_decay)));	
		last_error = error; 

		if(rmse <= m_paramsLearning.epsilon)
			break;
/*
		if(last_rmse > 0.0)
		{
			if(fabs(rmse - last_rmse) * 10000.0 < last_rmse)
				break; 
		}
*/
		last_rmse = rmse;  
	}
	printf("** training done! time cost %.3f sec\n", timer.GetTotal_asSec());
}


int32_t FMSNN::Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len)
{
	if(!y || !x)
		return _METIS_NN_ERROR_INPUT_NULL;
	if(y_len != m_paramsFMSNN.output || x_len != m_paramsFMSNN.input - 1)
		return _METIS_NN_ERROR_WRONG_LEN;
	//if(m_wf.IsNull() || !m_vfs || !m_whs || m_wo.IsNull())
	if(!m_vfs || !m_whs || m_wo.IsNull())
		return _METIS_NN_ERROR_MODEL_NULL;

	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();	// number of hidden layers
	int32_t fm_layer = m_paramsFMSNN.fm_inter + m_paramsFMSNN.input; 
	// for thread save, do not use inner layer
	double* ai = new double[m_paramsFMSNN.input];		// input layer
	double* af = new double[fm_layer];	// FM layer
	double** ahs = new double*[hl];			// hidden layers

	// activate input layer
	for(int32_t i = 0; i < x_len; i++) 
		ai[i] = x[i]; 
	ai[x_len] = 1.0;	// set bias

	// activate FM layer
	//ActivateFMLayer(af, m_paramsFMSNN.fm_layer, m_paramsFMSNN.fm_inter, ai, m_paramsFMSNN.input, m_wf, m_vfs, m_paramsFMSNN.fm_k, m_paramsFMSNN.act_fm_layer); 
	ActivateFMLayer(af, m_paramsFMSNN.fm_inter, ai, m_paramsFMSNN.input, m_vfs, m_paramsFMSNN.fm_k, m_paramsFMSNN.act_fm_layer); 

	// activate hiddens layer-by-layer
	for(int32_t h = 0; h < hl; h++) 
	{
		ahs[h] = new double[m_paramsFMSNN.vtr_hidden[h]]; 
		if(h == 0)
			ActivateForward(ahs[h], m_paramsFMSNN.vtr_hidden[h], af, fm_layer, m_whs[h], m_paramsFMSNN.act_hidden); 
		else
			ActivateForward(ahs[h], m_paramsFMSNN.vtr_hidden[h], ahs[h-1], m_paramsFMSNN.vtr_hidden[h-1], m_whs[h], m_paramsFMSNN.act_hidden); 
	}

	// activate output layer
	ActivateForward(y, y_len, ahs[hl-1], m_paramsFMSNN.vtr_hidden[hl-1], m_wo, m_paramsFMSNN.act_output);  

	delete [] ai; 
	delete [] af; 
	for(int32_t h = 0; h < hl; h++) 
		delete [] ahs[h];
	delete [] ahs;

	return _METIS_NN_SUCCESS; 	
}



int32_t FMSNN::Save(const char* sFile)
{
	//if(m_wf.IsNull() || !m_vfs || !m_whs || m_wo.IsNull())
	if(!m_vfs || !m_whs || m_wo.IsNull())
		return _METIS_NN_ERROR_MODEL_NULL; 
	ofstream ofs(sFile); 
	if(!ofs.is_open())
		return _METIS_NN_ERROR_FILE_OPEN;

	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();	// number of hidden layers

	ofs<<"** FM Supported Neural Network **"<<endl; 
	ofs<<endl;

	// save learning parameters
	ofs<<"@learning_params"<<endl; 
	TypeDefs::Print_PerceptronLearningParamsT(ofs, m_paramsLearning); 
	ofs<<endl; 

	// save architecture parameters of FMSNN 
	ofs<<"@architecture_params"<<endl; 
	TypeDefs::Print_FMSNNParamsT(ofs, m_paramsFMSNN); 
	ofs<<endl; 

	// save weight matrix of FM layer
//	ofs<<"@weight_fmlayer"<<endl; 
//	Matrix::Print_Matrix(ofs, m_wf);
//	ofs<<endl; 

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
	return _METIS_NN_SUCCESS; 
}


int32_t FMSNN::Load(const char* sFile)
{
	ifstream ifs(sFile);  
	if(!ifs.is_open())
		return _METIS_NN_ERROR_FILE_OPEN;
	Release(); 

	string str; 
	int32_t idx; 

	std::getline(ifs, str); 
	if(str != "** FM Supported Neural Network **")
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
			if(!TypeDefs::Read_PerceptronLearningParamsT(m_paramsLearning, ifs))
				return _METIS_NN_ERROR_LERANING_PARAMS;
		}
		else if(str == "@architecture_params")
		{
			if(!TypeDefs::Read_FMSNNParamsT(m_paramsFMSNN, ifs))
				return _METIS_NN_ERROR_ACH_PARAMS;
			Create(); 
		}
//		else if(str == "@weight_fmlayer")
//		{
//			if(!Matrix::Read_Matrix(m_wf, ifs))
//				return _METIS_NN_ERROR_MODEL_DATA;
//		}
		else if(str.find("@interaction_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 
			if(!Matrix::Read_Matrix(m_vfs[idx], ifs))
				return _METIS_NN_ERROR_MODEL_DATA;
		}
		else if(str.find("@weight_hidden_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 
			if(!Matrix::Read_Matrix(m_whs[idx], ifs))
				return _METIS_NN_ERROR_MODEL_DATA;
		}
		else if(str == "@weight_output")
		{
			if(!Matrix::Read_Matrix(m_wo, ifs))
				return _METIS_NN_ERROR_MODEL_DATA;
		}
	}

	ifs.close(); 
	return _METIS_NN_SUCCESS; 
}


FMSNNLearningParamsT FMSNN::GetLearningParams()
{
	return m_paramsLearning; 
}


FMSNNParamsT FMSNN::GetArchParams()
{
	return m_paramsFMSNN; 
}


bool FMSNN::SetByModelString(const char* sModelStr)
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
//		else if(ar.GetString(0) == "@wf")
//		{
//			if(!m_wf.FromString(ar.GetString(1).c_str()))
//				return false; 	
//		}
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


string FMSNN::ConvToModelString()
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


bool FMSNN::CombineWith(FMSNN& fms_nn, const double w0, const double w1)
{
	if(!TypeDefs::IsEqual_FMSNNParamsT(m_paramsFMSNN, fms_nn.GetArchParams()))
		return false; 

//	if(!m_wf.CombineWith(fms_nn.m_wf, w0, w1))
//		return false; 
	
	for(int32_t j = 0; j < m_paramsFMSNN.fm_inter; j++) 
	{
		if(!m_vfs[j].CombineWith(fms_nn.m_vfs[j], w0, w1))
			return false; 	
	}

	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();    // number of hidden layers	
	for(int32_t h = 0; h < hl; h++) 
	{
		if(!m_whs[h].CombineWith(fms_nn.m_whs[h], w0, w1))
			return false; 	
	}
		
	if(!m_wo.CombineWith(fms_nn.m_wo, w0, w1))
		return false; 

	return true; 
}


void FMSNN::ModelUpdate(const double learning_rate, const ERegula eRegula, const int32_t nPattCnt)
{
	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();	// number of hidden layers

	// update the transform matrix of output layer (m_wo)
	for(int32_t i = 0; i < m_paramsFMSNN.vtr_hidden[hl-1]; i++) 
	{
		for(int32_t j = 0; j < m_paramsFMSNN.output; j++) 
		{
			m_wo[i][j] -= learning_rate * (m_co[i][j] / (double)nPattCnt + Activation::DActRegula(m_wo[i][j], eRegula)); 
			m_co[i][j] = 0.0; 
		}
	}

	// update the transform matrices of hidden layers (m_whs)
	for(int32_t h = hl - 1; h > 0; h--)
	{
		for(int32_t i = 0; i < m_paramsFMSNN.vtr_hidden[h-1]; i++)
		{
			for(int32_t j = 0; j < m_paramsFMSNN.vtr_hidden[h]; j++)
			{
				m_whs[h][i][j] -= learning_rate * (m_chs[h][i][j] / (double)nPattCnt + Activation::DActRegula(m_whs[h][i][j], eRegula)); 
				m_chs[h][i][j] = 0.0; 
			}
		}
	}
	int32_t fm_layer = m_paramsFMSNN.fm_inter + m_paramsFMSNN.input; 
	for(int32_t i = 0; i < fm_layer; i++) 
	{
		for(int32_t j = 0; j < m_paramsFMSNN.vtr_hidden[0]; j++)
		{
			m_whs[0][i][j] -= learning_rate * (m_chs[0][i][j] / (double)nPattCnt + Activation::DActRegula(m_whs[0][i][j], eRegula)); 
			m_chs[0][i][j] = 0.0; 
		}
	}

	// update weight & interaction matrices of FM layer (m_wf & m_vfs)
	/*
	for(int32_t i = 0; i < m_paramsFMSNN.input; i++) 
	{
		for(int32_t j = 0; j < m_paramsFMSNN.fm_layer - m_paramsFMSNN.fm_inter; j++) 
		{
			m_wf[i][j] -= learning_rate * (m_cf[i][j] / (double)nPattCnt + Activation::DActRegula(m_wf[i][j], eRegula)); 
			m_cf[i][j] = 0.0; 
		}
	}
	*/
	for(int32_t j = 0; j < m_paramsFMSNN.fm_inter; j++) 
	{
		for(int32_t i = 0; i < m_paramsFMSNN.input - 1; i++) 
		{
			for(int32_t k = 0; k < m_paramsFMSNN.fm_k; k++) 
			{
				m_vfs[j][i][k] -= learning_rate * (m_cvfs[j][i][k] / (double)nPattCnt + Activation::DActRegula(m_vfs[j][i][k], eRegula)); 
				m_cvfs[j][i][k] = 0.0; 
			}
		}	
	}
}



////////////////////////////////////////////////////////////////////////////////////////////
// Internal Operations

void FMSNN::Create()
{
	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();	// number of hidden layers

//	m_wf.Create(m_paramsFMSNN.input, m_paramsFMSNN.fm_layer - m_paramsFMSNN.fm_inter);
//	Activation::InitTransformMatrix(m_wf, m_paramsFMSNN.act_fm_layer); 

	m_vfs = new Matrix[m_paramsFMSNN.fm_inter];
	for(int32_t j = 0; j < m_paramsFMSNN.fm_inter; j++) 
	{
		m_vfs[j].Create(m_paramsFMSNN.input - 1, m_paramsFMSNN.fm_k); 
		Activation::InitTransformMatrix(m_vfs[j], m_paramsFMSNN.act_fm_layer); 
	}

	int32_t fm_layer = m_paramsFMSNN.fm_inter + m_paramsFMSNN.input; 
	m_whs = new Matrix[hl]; 
	for(int32_t h = 0; h < hl; h++) 
	{
		if(h == 0)
			m_whs[h].Create(fm_layer, m_paramsFMSNN.vtr_hidden[h]);
		else
			m_whs[h].Create(m_paramsFMSNN.vtr_hidden[h-1], m_paramsFMSNN.vtr_hidden[h]);
		Activation::InitTransformMatrix(m_whs[h], m_paramsFMSNN.act_hidden); 
	}

	m_wo.Create(m_paramsFMSNN.vtr_hidden[hl-1], m_paramsFMSNN.output); 
	Activation::InitTransformMatrix(m_wo, m_paramsFMSNN.act_output); 
}


void FMSNN::Release()
{
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
	ReleaseAssistant();
}


void FMSNN::CreateAssistant()
{
	ReleaseAssistant();

	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();	// number of hidden layers
	int32_t fm_layer = m_paramsFMSNN.fm_inter + m_paramsFMSNN.input; 

	// create each layer
	m_ai = new double[m_paramsFMSNN.input]; 
	m_af = new double[fm_layer]; 
	m_ahs = new double*[hl]; 
	for(int32_t h = 0; h < hl; h++) 
		m_ahs[h] = new double[m_paramsFMSNN.vtr_hidden[h]]; 
	m_ao = new double[m_paramsFMSNN.output];

	// create delta arrays
	m_df = new double[m_paramsFMSNN.fm_inter]; 
	m_dhs = new double*[hl]; 
	for(int32_t h = 0; h < hl; h++) 
		m_dhs[h] = new double[m_paramsFMSNN.vtr_hidden[h]]; 
	m_do = new double[m_paramsFMSNN.output];

	// create change matrices
//	m_cf.Create(m_paramsFMSNN.input, m_paramsFMSNN.fm_layer - m_paramsFMSNN.fm_inter); 
//	m_cf.Init(0.0); 
	m_cvfs = new Matrix[m_paramsFMSNN.fm_inter];
	for(int32_t j = 0; j < m_paramsFMSNN.fm_inter; j++) 
	{
		m_cvfs[j].Create(m_paramsFMSNN.input - 1, m_paramsFMSNN.fm_k); 
		m_cvfs[j].Init(0.0); 
	}
	m_chs = new Matrix[hl];
	for(int32_t h = 0; h < hl; h++) 
	{
		if(h == 0)	
			m_chs[h].Create(fm_layer, m_paramsFMSNN.vtr_hidden[h]); 
		else	
			m_chs[h].Create(m_paramsFMSNN.vtr_hidden[h-1], m_paramsFMSNN.vtr_hidden[h]); 
		m_chs[h].Init(0.0); 
	}
	m_co.Create(m_paramsFMSNN.vtr_hidden[hl-1], m_paramsFMSNN.output); 
	m_co.Init(0.0); 
}


void FMSNN::ReleaseAssistant()
{
	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();	// number of hidden layers

	if(m_ai)
	{
		delete [] m_ai;
		m_ai = NULL; 
	}
	if(m_af)
	{
		delete [] m_af;
		m_af = NULL; 
	}
	if(m_ahs)
	{
		for(int32_t h = 0; h < hl; h++)
			delete [] m_ahs[h];
		delete [] m_ahs; 
		m_ahs = NULL; 
	}
	if(m_ao)
	{
		delete [] m_ao; 
		m_ao = NULL; 
	}

	if(m_df)
	{
		delete [] m_df; 
		m_df = NULL; 
	}
	if(m_dhs)
	{
		for(int32_t h = 0; h < hl; h++)
			delete [] m_dhs[h]; 
		delete [] m_dhs; 
		m_dhs = NULL;
	}
	if(m_do)
	{
		delete [] m_do; 
		m_do = NULL;
	}

	if(m_cvfs)
	{
		delete [] m_cvfs; 
		m_cvfs = NULL; 
	}
	if(m_chs)
	{
		delete [] m_chs; 
		m_chs = NULL; 
	}
}


void FMSNN::FeedForward(const double* in_vals, const int32_t in_len)
{
	if(!in_vals || in_len != m_paramsFMSNN.input - 1)
		throw "FMSNN::FeedForward() ERROR: Wrong length of \'in_vals\'!"; 

	// activate input layer
	for(int32_t i = 0; i < in_len; i++) 
		m_ai[i] = in_vals[i];
	m_ai[in_len] = 1.0;		// set bias

	// activate FM layer
	//ActivateFMLayer(m_af, m_paramsFMSNN.fm_layer, m_paramsFMSNN.fm_inter, m_ai, m_paramsFMSNN.input, m_wf, m_vfs, m_paramsFMSNN.fm_k, m_paramsFMSNN.act_fm_layer); 
	ActivateFMLayer(m_af, m_paramsFMSNN.fm_inter, m_ai, m_paramsFMSNN.input, m_vfs, m_paramsFMSNN.fm_k, m_paramsFMSNN.act_fm_layer); 

	// activate hiddens layer-by-layer
	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();	// number of hidden layers
	int32_t fm_layer = m_paramsFMSNN.fm_inter + m_paramsFMSNN.input; 
	for(int32_t h = 0; h < hl; h++) 
	{
		if(h == 0)
			ActivateForward(m_ahs[h], m_paramsFMSNN.vtr_hidden[h], m_af, fm_layer, m_whs[h], m_paramsFMSNN.act_hidden); 
		else
			ActivateForward(m_ahs[h], m_paramsFMSNN.vtr_hidden[h], m_ahs[h-1], m_paramsFMSNN.vtr_hidden[h-1], m_whs[h], m_paramsFMSNN.act_hidden); 
	}

	// activate output layer
	ActivateForward(m_ao, m_paramsFMSNN.output, m_ahs[hl-1], m_paramsFMSNN.vtr_hidden[hl-1], m_wo, m_paramsFMSNN.act_output);  
}


double FMSNN::BackPropagate(const double* out_vals, const int32_t out_len)
{
	if(!out_vals || out_len != m_paramsFMSNN.output)
		throw "FMSNN::BackPropagate() ERROR: Wrong length of \'out_vals\'!"; 

	double error = 0.0; 
	int32_t hl = (int32_t)m_paramsFMSNN.vtr_hidden.size();	// number of hidden layers

	// caculate delta and error of output layer
	for(int32_t j = 0; j < m_paramsFMSNN.output; j++) 
	{
		m_do[j] = m_ao[j] - out_vals[j]; 
		error += m_do[j] * m_do[j];  
	}

	// delta back propagate
	// to the top hidden
	for(int32_t i = 0; i < m_paramsFMSNN.vtr_hidden[hl-1]; i++)
	{
		m_dhs[hl-1][i] = 0.0;   
		for(int32_t j = 0; j < m_paramsFMSNN.output; j++)
			m_dhs[hl-1][i] += m_do[j] * m_wo[i][j]; 
	}
	// to lower hiddens
	for(int32_t h = hl-2; h >= 0; h--)
	{
		for(int32_t i = 0; i < m_paramsFMSNN.vtr_hidden[h]; i++) 
		{
			m_dhs[h][i] = 0.0;   
			for(int32_t j = 0; j < m_paramsFMSNN.vtr_hidden[h+1]; j++) 	
				m_dhs[h][i] += m_dhs[h+1][j] * m_whs[h+1][i][j];  
		}
	}
	// to FM layer
	for(int32_t i = 0; i < m_paramsFMSNN.fm_inter; i++) 
	{
		m_df[i] = 0.0;  	
		for(int32_t j = 0; j < m_paramsFMSNN.vtr_hidden[0]; j++) 
			m_df[i] += m_dhs[0][j] * m_whs[0][i][j];  	
	}

	// update change matrices
	// change matrix of output (m_co) 
	for(int32_t j = 0; j < m_paramsFMSNN.output; j++)
	{
		for(int32_t i = 0; i < m_paramsFMSNN.vtr_hidden[hl-1]; i++)
			m_co[i][j] += m_do[j] * Activation::DActivate(m_ao[j], m_paramsFMSNN.act_output) * m_ahs[hl-1][i];  
	}
	// change matrices of hiddens (m_chs) 
	for(int32_t h = hl - 1; h > 0; h--)
	{
		for(int32_t j = 0; j < m_paramsFMSNN.vtr_hidden[h]; j++)
		{
			for(int32_t i = 0; i < m_paramsFMSNN.vtr_hidden[h-1]; i++)
				m_chs[h][i][j] += m_dhs[h][j] * Activation::DActivate(m_ahs[h][j], m_paramsFMSNN.act_hidden) * m_ahs[h-1][i]; 	
		}	
	}
	int32_t fm_layer = m_paramsFMSNN.fm_inter + m_paramsFMSNN.input; 
	for(int32_t j = 0; j < m_paramsFMSNN.vtr_hidden[0]; j++)
	{
		for(int32_t i = 0; i < fm_layer; i++) 
			m_chs[0][i][j] += m_dhs[0][j] * Activation::DActivate(m_ahs[0][j], m_paramsFMSNN.act_hidden) * m_af[i]; 
	}
	// change matrices of weight & interactions in FM layer (m_cf & m_cvfs)
	double sum; 
	for(int32_t j = 0; j < m_paramsFMSNN.fm_inter; j++) 
	{ // m_cfvs
		for(int32_t k = 0; k < m_paramsFMSNN.fm_k; k++) 
		{
			sum = 0.0; 	
			for(int32_t i = 0; i < m_paramsFMSNN.input - 1; i++)  
				sum += m_vfs[j][i][k] * m_ai[i]; 
			for(int32_t i = 0; i < m_paramsFMSNN.input - 1; i++) 
				m_cvfs[j][i][k] += m_df[j] * Activation::DActivate(m_af[j], m_paramsFMSNN.act_fm_layer) * (m_ai[i] * sum - m_vfs[j][i][k] * m_ai[i] * m_ai[i]);
		}
	}	
	
	return error / double(m_paramsFMSNN.output); 
}


void FMSNN::ActivateFMLayer(double* af, const int32_t inter_size, const double* ai, const int32_t ai_size, 
		Matrix* vfs, const int32_t fm_k, const EActType act_type)
{
	double sum, inter_sum1, inter_sum2; 

	// interation (except bias)
	for(int32_t j = 0; j < inter_size; j++) 
	{
		sum = 0.0; 
		for(int32_t k = 0; k < fm_k; k++) 
		{
			inter_sum1 = 0.0; 
			inter_sum2 = 0.0; 
			for(int32_t i = 0; i < ai_size - 1; i++) 
			{
				inter_sum1 += vfs[j][i][k] * ai[i];  
				inter_sum2 += vfs[j][i][k] * ai[i] * vfs[j][i][k] * ai[i];  
			}
			sum += (inter_sum1 * inter_sum1 - inter_sum2) / 2.0; 	
		}
		af[j] = Activation::Activate(sum, act_type); 
	}
	
	for(int32_t i = 0; i < ai_size; i++) 
		af[i + inter_size] = ai[i]; 
}


void FMSNN::ActivateForward(double* up_a, const int32_t up_size, const double* low_a, const int32_t low_size, 
		Matrix& w, const EActType up_act_type)
{
	double e = 0.0, sum;

	for(int32_t j = 0; j < up_size; j++) 
	{
		sum = 0.0; 
		for(int32_t i = 0; i < low_size; i++) 
			sum += w[i][j] * low_a[i];
		// activation
		up_a[j] = Activation::Activate(sum, up_act_type); 
		// special for softmax
		if(up_act_type == _ACT_SOFTMAX)
			e += up_a[j];
	}
	// for softmax
	if(up_act_type == _ACT_SOFTMAX)
	{
		for(int32_t j = 0; j < up_size; j++) 
			up_a[j] /= e; 
	}
}


pair<double, double> FMSNN::Validation(vector<Pattern*>& vtrPatts, const int32_t nBackCnt)
{
	double error = 0.0; 
	int32_t correct = 0, total = 0; 
	int32_t patts = (int32_t)vtrPatts.size(); 
	double* y = new double[m_paramsFMSNN.output];
	int32_t k; 

	for(int32_t i = 0; i < nBackCnt && i < patts; i++) 
	{
		k = patts-1-i;
		Predict(y, m_paramsFMSNN.output, vtrPatts[k]->m_x, vtrPatts[k]->m_nXCnt); 
		if(Pattern::MaxOff(y, m_paramsFMSNN.output) == Pattern::MaxOff(vtrPatts[k]->m_y, vtrPatts[k]->m_nYCnt))
			correct += 1; 	
		error += Pattern::Error(y, vtrPatts[k]->m_y, m_paramsFMSNN.output);  	
		total += 1; 	
	}

	delete [] y;

	double pr = (double)correct / (double)total;
	double rmse = sqrt(error / (double)total);

	return pair<double,double>(pr, rmse); 
}



