#include "NeuralNetwork.h"
using namespace metis_nn; 
#include "Timer.h"
#include "RocAnalyzer.h"
using namespace metis_uti; 
#include <algorithm>
using namespace std; 
#include <math.h>
#include <stdio.h>


bool NeuralNetwork::m_bUpdateCancel = false;


NeuralNetwork::NeuralNetwork(const ENNType eNNType) : m_eNNType(eNNType), m_pLearnParams(NULL), m_pArchParams(NULL) 
{
}


NeuralNetwork::~NeuralNetwork()
{
	if(m_pLearnParams)
	{
		delete m_pLearnParams; 
		m_pLearnParams = NULL;
	}
	if(m_pArchParams)
	{
		delete m_pArchParams;
		m_pArchParams = NULL; 
	}
}


bool NeuralNetwork::Train(vector<Pattern*>& vtrPatts)
{
	if(IsNull())
	{
		printf("Error, the Model hasn't been initialized correctly!\n");
		return false;  
	}
	NNAssi** pp_assi = CreateAssi(true); 
	if(!pp_assi)
	{
		printf("Error, failed to create assistant parameters for training!\n"); 
		return false;  
	}
	double learning_rate = m_pLearnParams->p_optim_params->learning_rate_init;	// learning rate 
	int32_t ss = 0; 
	double avg_loss, mini_loss = 99999999999.9;	// loss of one epoch
	Timer timer;		// timer

	if(m_pLearnParams->early_stop == 0)
	{ // early stop is disabled
		for(int32_t t = 0; t < m_pLearnParams->max_epoches; t++) 
		{
			timer.Start(); 	
			// learning_rate /= sqrt((double)t + 1.0); 
			if(!Epoch(avg_loss, pp_assi, vtrPatts, learning_rate)) 
			{
				ReleaseAssi(pp_assi); 
				printf("epoch %d | failed!!\n", t+1); 
				return false; 
			}
			ss++; 	
			if(avg_loss < mini_loss)
			{
				mini_loss = avg_loss; 
				ss = 0; 		
			}
			timer.Stop(); 	
			printf("epoch %d | training_loss: %.12g, ss: %d | time_cost(s): %.3f\n", 
					t+1, avg_loss, ss, timer.GetLast_asSec()); 
			if(ss >= 10 || avg_loss < m_pLearnParams->epsilon)
				break; 	
		}
	}
	else
	{ // early stop is enabled
		// shuffle patterns
		random_shuffle(vtrPatts.begin(), vtrPatts.end());
		// select patterns for testing & training
		int32_t test_cnt = (int32_t)vtrPatts.size() / 10 < 1000 ? (int32_t)vtrPatts.size() / 10 : 1000;  

		int32_t s = 0; 
		double validated_loss, mini_validated_loss = 99999999999.9;
		double auc;  
		for(int32_t t = 0; t < m_pLearnParams->max_epoches; t++) 
		{
			timer.Start();

			// epoch 	
			// learning_rate /= sqrt((double)t + 1.0); 
			if(!Epoch(avg_loss, pp_assi, vtrPatts, learning_rate, test_cnt)) 
			{
				ReleaseAssi(pp_assi); 
				printf("epoch %d | failed!!\n", t+1); 
				return false; 
			}
			ss++; 	
			if(avg_loss < mini_loss)
			{
				mini_loss = avg_loss; 
				ss = 0; 		
			}
			timer.Stop(); 	

			// validation
			if(vtrPatts[0]->m_nYCnt == 2)
			{
				Validation_Binary(validated_loss, auc, vtrPatts, test_cnt); 
				s++; 	
				if(validated_loss < mini_validated_loss)
				{
					mini_validated_loss = validated_loss; 
					s = 0; 		
				}
				printf("epoch %d | training_loss: %.12g, ss: %d | validated_loss: %.12g, auc: %.6g, s: %d | time_cost(s): %.3f\n", 
						t+1, avg_loss, ss, validated_loss, auc, s, timer.GetLast_asSec()); 
			}
			else
			{
				validated_loss = Validation(vtrPatts, test_cnt); 
				s++; 	
				if(validated_loss < mini_validated_loss)
				{
					mini_validated_loss = validated_loss; 
					s = 0; 		
				}
				printf("epoch %d | training_loss: %.12g, ss: %d | validated_loss: %.12g, s: %d | time_cost(s): %.3f\n", 
						t+1, avg_loss, ss, validated_loss, s, timer.GetLast_asSec()); 
			}

			if(s >= m_pLearnParams->early_stop || ss >= m_pLearnParams->early_stop || avg_loss <= m_pLearnParams->epsilon)
				break; 	
		}
	}
	printf("** training done! time cost %.3f sec\n", timer.GetTotal_asSec());

	ReleaseAssi(pp_assi); 
	return true; 
}


ENNType NeuralNetwork::GetNNType()
{
	return m_eNNType; 
}


LearnParams* NeuralNetwork::GetLearnParams() const
{
	return m_pLearnParams; 
}


ArchParams* NeuralNetwork::GetArchParams() const
{
	return m_pArchParams; 
} 	


bool NeuralNetwork::SetCancelFlag(const bool bCancel)
{
	m_bUpdateCancel = bCancel;  
	return m_bUpdateCancel; 
}


void NeuralNetwork::PrintParams(ostream& os)
{
	if(m_pLearnParams)
		m_pLearnParams->Print(os); 
	else
		os<<"LEARN_PARAMS:null"<<endl; 
	
	if(m_pArchParams)
		m_pArchParams->Print(os); 
	else
		os<<"ARCH_PARAMS:null"<<endl; 
} 


EActType NeuralNetwork::OutputActType(const int32_t nOutput)
{
	if(nOutput < 2)		// regression
		return _ACT_LINEAR; 
	else if(nOutput == 2)	// bi-classification
		return _ACT_SIGMOID; 
	else			// multi-classification
		return _ACT_SOFTMAX; 
}


void NeuralNetwork::WeightMatrixInit(Matrix& w)
{
	double a = 4.0 * sqrt(6.0 / (double)(w.Rows() + w.Cols()));
	w.Init_RandUni(0.0 - a, a); 
}


int32_t NeuralNetwork::Validation_Binary(double& dValidatedLoss, double& dAuc, vector<Pattern*>& vtrValidation, const int32_t nMaxCnt)
{
	if(vtrValidation.empty())
		return 0.0; 
	int32_t max_cnt = (nMaxCnt < (int32_t)vtrValidation.size() && nMaxCnt > 0)? nMaxCnt : (int32_t)vtrValidation.size();

	int32_t cnt = 0; 
	int32_t pred_len = vtrValidation[0]->m_nYCnt; 
	double* pred = new double[pred_len];  
	RocAnalyzer roc; 

	dValidatedLoss = 0.0; 
	dAuc = 0.5; 

	for(int32_t k = 0; k < max_cnt; k++) 
	{
		int32_t ret = Predict(pred, pred_len, vtrValidation[k]->m_x, vtrValidation[k]->m_nXCnt); 
		if(ret == _METIS_NN_SUCCESS)
		{
			if(m_pArchParams->output == 1)  // regression, quadratic as loss
				dValidatedLoss += Activation::Loss_Quadratic(pred[0], vtrValidation[k]->m_y[0]); 
			else if(m_pArchParams->output == 2) // bi-classification, cross entropy as loss
				dValidatedLoss += Activation::Loss_CrossEntropy(pred[0], vtrValidation[k]->m_y[0]); 
			else	// multi-classification, log likelihood as loss
				dValidatedLoss += Activation::Loss_LogLikelihood(pred, vtrValidation[k]->m_y, pred_len); 

			if(Pattern::MaxOff(vtrValidation[k]->m_y, vtrValidation[k]->m_nYCnt) == 0)
				roc.Insert(_POSITIVE, pred[0]); 
			else 
				roc.Insert(_NEGATIVE, pred[0]); 

			cnt++; 
		}
	}

	delete pred; 
	dAuc = roc.Auc(); 
	dValidatedLoss /= (double)cnt; 
	return cnt; 
}

 
double NeuralNetwork::Validation(vector<Pattern*>& vtrValidation, const int32_t nMaxCnt)
{
	if(vtrValidation.empty())
		return 0.0; 
	int32_t max_cnt = (nMaxCnt < (int32_t)vtrValidation.size() && nMaxCnt > 0)? nMaxCnt : (int32_t)vtrValidation.size();

	int32_t cnt = 0; 
	int32_t pred_len = vtrValidation[0]->m_nYCnt; 
	double* pred = new double[pred_len];  
	double loss = 0.0; 

	for(int32_t k = 0; k < max_cnt; k++) 
	{
		int32_t ret = Predict(pred, pred_len, vtrValidation[k]->m_x, vtrValidation[k]->m_nXCnt); 
		if(ret == _METIS_NN_SUCCESS)
		{
			if(m_pArchParams->output == 1)	// regression, quadratic as loss
				loss += Activation::Loss_Quadratic(pred[0], vtrValidation[k]->m_y[0]); 
			else if(m_pArchParams->output == 2) // bi-classification, cross entropy as loss
				loss += Activation::Loss_CrossEntropy(pred[0], vtrValidation[k]->m_y[0]); 
			else	// multi-classification, log likelihood as loss
				loss += Activation::Loss_LogLikelihood(pred, vtrValidation[k]->m_y, pred_len); 
			cnt++; 	
		}
	}

	delete pred; 
	return loss / (double)cnt; 
}


