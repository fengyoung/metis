#include "FM.h"
#include "Activation.h"
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

FM::FM()
{
	m_vos = NULL; 
	m_ai = NULL; 
	m_ao = NULL; 
	m_do = NULL; 
	m_cvos = NULL; 
}


FM::~FM()
{
	Release(); 
}



////////////////////////////////////////////////////////////////////////////////////////////
// Operations 

void FM::Init(const FMParamsT fmParamsT, const FMLearningParamsT fmLearningParamsT)
{
	Release(); 

	m_paramsFM = fmParamsT;
	m_paramsFM.input += 1;	// add 1 for bias nodes 
	m_paramsLearning = fmLearningParamsT; 

	Create(); 
}


bool FM::InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	Release(); 

	m_paramsFM.input = nInput + 1;	// add 1 for bias nodes
	m_paramsFM.output = nOutput; 
	m_paramsFM.act_output = TypeDefs::ActType(conf.GetVal_asString("Activation").c_str());   
	if(m_paramsFM.act_output == _ACT_NONE)
		return false; 
	m_paramsFM.fm_k = conf.GetVal_asInt("FM_K"); 

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


void FM::Train(vector<Pattern*>& vtrPatts)
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


void FM::TrainAndValidate(vector<Pattern*>& vtrPatts)
{
	int32_t cross_cnt = (int32_t)vtrPatts.size() / 20;			// 5% patterns for corss validation
	int32_t train_cnt = (int32_t)vtrPatts.size() - cross_cnt;	// 95% patterns for training
	double learning_rate = m_paramsLearning.learning_rate;	// learning rate, it would be update after every iteration
	double last_error = -1.0, error, rmse, last_rmse = -1.0;	// training error and RMSE in one iteration
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


int32_t FM::Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len)
{
	if(!y || !x)
		return _METIS_NN_ERROR_INPUT_NULL;
	if(y_len != m_paramsFM.output || x_len != m_paramsFM.input - 1)
		return _METIS_NN_ERROR_WRONG_LEN;
	if(m_wo.IsNull() || !m_vos)
		return _METIS_NN_ERROR_MODEL_NULL;

	// for thread safe, do not use inner layer
	double* ai = new double[m_paramsFM.input];		// input layer

	// activate input layer
	for(int32_t i = 0; i < x_len; i++) 
		ai[i] = x[i];
	ai[x_len] = 1.0;		// set bias

	// activate output 
	double sum, inter_sum1, inter_sum2; 	
	double e = 0.0; 
	for(int32_t j = 0; j < m_paramsFM.output; j++)	
	{
		sum = 0.0; 	
		// dim effect
		for(int32_t i = 0; i < m_paramsFM.input; i++)
			sum += ai[i] * m_wo[i][j];
		// interaction effect
		for(int32_t k = 0; k < m_paramsFM.fm_k; k++) 
		{
			inter_sum1 = 0.0; 	
			inter_sum2 = 0.0; 	
			for(int32_t i = 0; i < m_paramsFM.input - 1; i++)
			{
				inter_sum1 += m_vos[j][i][k] * ai[i];
				inter_sum2 += m_vos[j][i][k] * m_vos[j][i][k] * ai[i] * ai[i]; 
			}
			sum += (inter_sum1 * inter_sum1 - inter_sum2) / 2.0; 
		}
		// output
		y[j] = Activation::Activate(sum, m_paramsFM.act_output);  
		// special for softmax
		if(m_paramsFM.act_output == _ACT_SOFTMAX)
			e += y[j]; 
	}
	// for softmax
	if(m_paramsFM.act_output == _ACT_SOFTMAX)
	{
		for(int32_t j = 0; j < m_paramsFM.output; j++)	
			y[j] /= e; 
	}

	delete [] ai; 

	return _METIS_NN_SUCCESS; 
}


int32_t FM::Save(const char* sFile)
{
	if(m_wo.IsNull() || !m_vos)
		return _METIS_NN_ERROR_MODEL_NULL; 
	ofstream ofs(sFile); 
	if(!ofs.is_open())
		return _METIS_NN_ERROR_FILE_OPEN;

	ofs<<"** Factorization Machine **"<<endl; 
	ofs<<endl;

	// save learning parameters
	ofs<<"@learning_params"<<endl; 
	TypeDefs::Print_PerceptronLearningParamsT(ofs, m_paramsLearning); 
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
		Matrix::Print_Matrix(ofs, m_vos[j]); 
		ofs<<endl; 
	}

	ofs.close(); 
	return _METIS_NN_SUCCESS; 
}


int32_t FM::Load(const char* sFile)
{
	ifstream ifs(sFile);  
	if(!ifs.is_open())
		return _METIS_NN_ERROR_FILE_OPEN;
	Release(); 

	string str; 
	int32_t idx; 

	std::getline(ifs, str); 
	if(str != "** Factorization Machine **")
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
			if(!TypeDefs::Read_FMParamsT(m_paramsFM, ifs))
				return _METIS_NN_ERROR_ACH_PARAMS;
			Create(); 
		}
		else if(str == "@weight")
		{
			if(!Matrix::Read_Matrix(m_wo, ifs))
				return _METIS_NN_ERROR_MODEL_DATA;
		}
		else if(str.find("@interaction_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 
			if(!Matrix::Read_Matrix(m_vos[idx], ifs))
				return _METIS_NN_ERROR_MODEL_DATA;
		}
	}

	ifs.close();
	return _METIS_NN_SUCCESS; 
}


FMLearningParamsT FM::GetLearningParams()
{
	return m_paramsLearning; 
}


FMParamsT FM::GetArchParams()
{
	return m_paramsFM; 
}


bool FM::SetByModelString(const char* sModelStr)
{
	if(!sModelStr)
		return false;
	StringArray array(sModelStr, "|"); 
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
			m_vos = new Matrix[m_paramsFM.output];
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
			//m_vos[idx].Create(m_paramsFM.input-1, m_paramsFM.fm_k); 
			if(!m_vos[idx].FromString(ar.GetString(1).c_str()))
				return false; 
		}
	}

	return true; 
}


string FM::ConvToModelString()
{
	char stmp[32];
	
	string str("fm");
	str += "|@ap:"; 
	str += TypeDefs::ToString_FMParamsT(m_paramsFM);  
	str += "|@w:"; 
	str += m_wo.ToString();
	for(int32_t j = 0; j < m_paramsFM.output; j++)
	{
		sprintf(stmp, "|@inter_%d:", j);
		str += stmp; 	
		str += m_vos[j].ToString(); 
	}
	return str; 
}


bool FM::CombineWith(FM& fm, const double w0, const double w1)
{
	if(!TypeDefs::IsEqual_FMParamsT(m_paramsFM, fm.GetArchParams()))
		return false; 
	if(!m_wo.CombineWith(fm.m_wo, w0, w1))
		return false; 
	for(int32_t j = 0; j < m_paramsFM.output; j++) 
	{
		if(!m_vos[j].CombineWith(fm.m_vos[j], w0, w1))
			return false; 
	}
	return true; 
}


////////////////////////////////////////////////////////////////////////////////////////////
// Internal Operations 

void FM::Create()
{
	m_wo.Create(m_paramsFM.input, m_paramsFM.output);
	Activation::InitTransformMatrix(m_wo, m_paramsFM.act_output); 
	m_vos = new Matrix[m_paramsFM.output]; 
	for(int32_t j = 0; j < m_paramsFM.output; j++) 
	{
		m_vos[j].Create(m_paramsFM.input-1, m_paramsFM.fm_k);
		Activation::InitTransformMatrix(m_vos[j], m_paramsFM.act_output); 
	}
}


void FM::Release()
{
	if(m_vos)
	{
		delete [] m_vos; 
		m_vos = NULL; 
	}
	ReleaseAssistant();
}


void FM::CreateAssistant()
{
	// create change matrices
	m_co.Create(m_paramsFM.input, m_paramsFM.output); 
	m_co.Init(0.0);
	m_cvos = new Matrix[m_paramsFM.input - 1];
	for(int32_t j = 0; j < m_paramsFM.output; j++) 
	{
		m_cvos[j].Create(m_paramsFM.input - 1, m_paramsFM.fm_k); 
		m_cvos[j].Init(0.0); 
	}

	// create input layer
	m_ai = new double[m_paramsFM.input];
	// create output layer
	m_ao = new double[m_paramsFM.output];

	// create delta array
	m_do = new double[m_paramsFM.output];
}


void FM::ReleaseAssistant()
{
	if(m_ai)
	{
		delete [] m_ai; 
		m_ai = NULL; 
	}
	if(m_ao)
	{
		delete [] m_ao; 
		m_ao = NULL; 
	}
	if(m_do)
	{
		delete [] m_do; 
		m_do = NULL; 
	}
	if(m_cvos)
	{
		delete [] m_cvos; 
		m_cvos = NULL; 
	}
}


void FM::FeedForward(const double* in_vals, const int32_t in_len)
{
	if(!in_vals || in_len != m_paramsFM.input - 1)
		throw "FM::FeedForward() ERROR: Wrong length of \'in_vals\'!"; 

	// activate input layer
	for(int32_t i = 0; i < in_len; i++) 
		m_ai[i] = in_vals[i];
	m_ai[in_len] = 1.0;		// set bias

	// activate output layer
	double sum, inter_sum1, inter_sum2; 	
	double e = 0.0; 
	for(int32_t j = 0; j < m_paramsFM.output; j++)	
	{
		// dim effect
		sum = 0.0; 	
		for(int32_t i = 0; i < m_paramsFM.input; i++)
			sum += m_ai[i] * m_wo[i][j];
		// interaction effect
		for(int32_t k = 0; k < m_paramsFM.fm_k; k++) 
		{
			inter_sum1 = 0.0; 	
			inter_sum2 = 0.0; 	
			for(int32_t i = 0; i < m_paramsFM.input - 1; i++)
			{
				inter_sum1 += m_vos[j][i][k] * m_ai[i];
				inter_sum2 += m_vos[j][i][k] * m_vos[j][i][k] * m_ai[i] * m_ai[i]; 
			}
			sum += (inter_sum1 * inter_sum1 - inter_sum2) / 2.0; 
		}
		// output
		m_ao[j] = Activation::Activate(sum, m_paramsFM.act_output);  
		// special for softmax
		if(m_paramsFM.act_output == _ACT_SOFTMAX)
			e += m_ao[j]; 
	}
	// for softmax
	if(m_paramsFM.act_output == _ACT_SOFTMAX)
	{
		for(int32_t j = 0; j < m_paramsFM.output; j++)	
			m_ao[j] /= e; 
	}
}


double FM::BackPropagate(const double* out_vals, const int32_t out_len)
{
	if(!out_vals || out_len != m_paramsFM.output)
		throw "FM::BackPropagate() ERROR: Wrong length of \'out_vals\'!"; 

	double error = 0.0; 
	// calculate delta and error of output 
	for(int32_t j = 0; j < m_paramsFM.output; j++) 
	{
		m_do[j] = m_ao[j] - out_vals[j]; 
		error += m_do[j] * m_do[j];  
	}

	// update change matrix
	double sum; 
	for(int32_t j = 0; j < m_paramsFM.output; j++)
	{ 
		// m_co
		for(int32_t i = 0; i < m_paramsFM.input; i++) 
			m_co[i][j] += m_do[j] * Activation::DActivate(m_ao[j], m_paramsFM.act_output) * m_ai[i]; 
		// m_cvos
		for(int32_t k = 0; k < m_paramsFM.fm_k; k++) 
		{
			sum = 0.0; 
			for(int32_t i = 0; i < m_paramsFM.input - 1; i++) 
				sum += m_vos[j][i][k] * m_ai[i]; 
			for(int32_t i = 0; i < m_paramsFM.input - 1; i++) 
				m_cvos[j][i][k] += m_do[j] * Activation::DActivate(m_ao[j], m_paramsFM.act_output) * (m_ai[i] * sum - m_vos[j][i][k] * m_ai[i] * m_ai[i]);
		}
	}

	return error / double(m_paramsFM.output); 
}


void FM::ModelUpdate(const double learning_rate, const ERegula eRegula, const int32_t nPattCnt)
{
	// update weight matrix
	for(int32_t i = 0; i < m_paramsFM.input; i++) 
	{
		for(int32_t j = 0; j < m_paramsFM.output; j++) 
		{
			m_wo[i][j] -= learning_rate * (m_co[i][j] / (double)nPattCnt + Activation::DActRegula(m_wo[i][j], eRegula)); 
			m_co[i][j] = 0.0; 
		}
	}

	// update interaction matrices
	for(int32_t j = 0; j < m_paramsFM.output; j++) 
	{
		for(int32_t i = 0; i < m_paramsFM.input - 1; i++) 
		{
			for(int32_t k = 0; k < m_paramsFM.fm_k; k++) 
			{
				m_vos[j][i][k] -= learning_rate * (m_cvos[j][i][k] / (double)nPattCnt + Activation::DActRegula(m_vos[j][i][k], eRegula)); 
				m_cvos[j][i][k] = 0.0; 
			}
		}	
	}
}


pair<double, double> FM::Validation(vector<Pattern*>& vtrPatts, const int32_t nBackCnt)
{
	double error = 0.0; 
	int32_t correct = 0, total = 0; 
	int32_t patts = (int32_t)vtrPatts.size(); 
	double* y = new double[m_paramsFM.output];
	int32_t k; 

	for(int32_t i = 0; i < nBackCnt && i < patts; i++) 
	{
		k = patts-1-i;
		Predict(y, m_paramsFM.output, vtrPatts[k]->m_x, vtrPatts[k]->m_nXCnt); 
		if(Pattern::MaxOff(y, m_paramsFM.output) == Pattern::MaxOff(vtrPatts[k]->m_y, vtrPatts[k]->m_nYCnt))
			correct += 1; 	
		error += Pattern::Error(y, vtrPatts[k]->m_y, m_paramsFM.output);  	
		total += 1; 	
	}

	delete [] y;

	double pr = (double)correct / (double)total;
	double rmse = sqrt(error / (double)total);

	return pair<double,double>(pr, rmse); 
}




