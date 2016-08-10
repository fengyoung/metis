// FMSNN.h
//
// Factorization Machine Supported Neural Network (FMSNN) consists one input layer, one output layer, one FM layer and some hidden layers. 
//
// In the FMSNN architecture, the number of input nodes equls the number of input signal add 1, the "1" for bias.
// FM layer get the interaction between each features
// The number of output nodes equls the number of output signal 
// The number of hidden layers and each layer node number are user defined
// Nodes on hidden and output layers take activation functions   
//
// IMPORTANT
// (1) when using the logit (sigmoid) activation function for the output layer make sure y values are scaled from 0 to 1
// (2) when using the tanh activation function for the output layer make sure y values are scaled from 0 to 1
// (3) sigmoid function is suggested in output layer for binary classification, and softmax is suggested for multiple classfication
// (4) the output activation can be set as 'none' for regression
// (5) tanh is recommended for FM layers
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v1.0 2016-03-14
//


#ifndef _METIS_NN_FMSNN_H
#define _METIS_NN_FMSNN_H

#include <string>
#include <vector>
using namespace std;
#include <stdint.h>
#include "TypeDefs.h"
#include "Pattern.h"
#include "Matrix.h"
#include "Activation.h"


namespace metis_nn
{

// CLASS
//	FMSNN - Factorization Machine Supported Neural Network
//
// DESCRIPTION
//	This FMSNN neural network is based on Back-Propagation (BP) algorithm. 
//	The training supports online(SGD), mini-batch(MSGD) and batch(GD) mode
//
class FMSNN
{
public:
	// Construction & Destruction
	FMSNN();
	virtual ~FMSNN();
	
	// NAME
	//	Init - initialize the FMSNN parameters, including learning parameters and FMSNN architecture parameters 
	//	InitFromConfig - initialize the FMSNN parameters by values read from config file
	// 
	// DESCRIPTION
	//	fmsnnParamsT: architecture parameters of FMSNN
	//	fmsnnLearningParamsT: learning parameters
	//	sConfigFile: config file
	//	nInput: number of input signal 
	//	nOnput: number of output signal 
	//
	// RETURN
	//	true for success, false for some errors
	void Init(const FMSNNParamsT fmsnnParamsT, const FMSNNLearningParamsT fmsnnLearningParamsT); 
	bool InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput);

	// NAME
	//	Train - train FMSNN model by patterns
	//	TrainAndValidate - train FMSNN model with validation	
	// 
	// DESCRIPTION
	//	vtrPatts: list of training patterns
	void Train(vector<Pattern*>& vtrPatts);	
	void TrainAndValidate(vector<Pattern*>& vtrPatts);	

	// NAME
	//	Predict - calculate prediction of input signal based on current FMSNN model
	// 
	// DESCRIPTION
	//	y: output parameter, the prediction result
	//	y_len: size of y
	//	x: input signal
	//	x_len: size of x
	// 
	// RETURN
	//	Return _METIS_NN_SUCCESS for success, others for some errors
	int32_t Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len); 

	// NAME
	//	Save - save current FMSNN model to file
	//	Load - load FMSNN model from file to construct current object
	//
	// DESCRIPTION
	//	sFile - FMSNN model file
	// 
	// RETURN
	//	Return _METIS_NN_SUCCESS for success, others for some errors
	int32_t Save(const char* sFile); 
	int32_t Load(const char* sFile); 

	// Get learning parameters
	FMSNNLearningParamsT GetLearningParams(); 
	
	// Get architecture parameters 
	FMSNNParamsT GetArchParams(); 
	
	// Set current FMSNN by model string
	bool SetByModelString(const char* sModelStr); 
	// Convert current FMSNN to model string
	string ConvToModelString(); 	
	// Combine current FMSNN which another model
	bool CombineWith(FMSNN& fms_nn, const double w0, const double w1);

	// NAME
	//	ModelUpdate - update all transform matrices in FMSNN 
	// 
	// DESCRIPTION
	//	After transform matrices updating, elements of all change matrices should be set as 0
	//	
	//	learning_rate: learning rate
	//	eRegula: regularzation type
	//	nPattCnt: count of training patterns	
	void ModelUpdate(const double learning_rate, const ERegula eRegula, const int32_t nPattCnt);


protected:
	// Create model variables, allocate menory for them 
	void Create();
	// Release inner objects
	void Release();
	// Create assistant variables, allocate menory for them 
	void CreateAssistant();
	// Release assistant variables
	void ReleaseAssistant(); 

	// NAME
	//	FeedForward - forward phase
	// 
	// DESCRIPTION
	//	in_vals: input signal
	//	in_len: size of input signal
	void FeedForward(const double* in_vals, const int32_t in_len); 
	
	// NAME
	//	BackPropagate - backword phase
	//	
	// DESCRIPTION
	//	out_vals: output signal
	//	out_len: size of output signal
	//
	// RETURN
	//	The error (difference) between output layer and output signal 
	double BackPropagate(const double* out_vals, const int32_t out_len); 

	// NAME
	//	ActivateFMLayer - activate FM layer by input 
	//
	// DESCRIPTION
	//	af: FM layer
	//	ai: input layer
	//	ai_size: size of input layer
	//	inter_size: size of interactions 
	//	vfs: interaction matirces 
	//	fm_k: K value of FM
	//  act_type: activation function type of FM layer
	void ActivateFMLayer(double* af, const int32_t inter_size, const double* ai, const int32_t ai_size, 
			Matrix* vfs, const int32_t fm_k, const EActType act_type); 
	
	// NAME
	//	ActivateForward - forward activation, activate upper layer by lower layer
	//
	// DESCRIPTION
	//	up_a: out parameter, upper layer
	//	up_size: size of upper layer
	//	low_a: lower layer
	//	low_size: size of lower layer
	//	w: transform matrix
	//	up_act_type: activation function type of upper layer
	void ActivateForward(double* up_a, const int32_t up_size, const double* low_a, const int32_t low_size, 
			Matrix& w, const EActType up_act_type); 
	
	// NAME
	//	Validation - validate current FMSNN model
	// 
	// DESCRIPTION	
	//	vtrPatts: training patterns list
	//	nBackCnt: number of validation patterns, which are at back of the list
	// 
	// RETRUN
	//	The precision and RMSE
	pair<double, double> Validation(vector<Pattern*>& vtrPatts, const int32_t nBackCnt);

public: 
	// model variables
//	Matrix m_wf;	// weight matrix of FM layer ((ni+1) * nfm)
	Matrix* m_vfs;	// interaction matirces of FM layer (nfm * ni * fm_k)
	Matrix* m_whs;	// transform matrices of hidden layers
	Matrix m_wo;	// transform matrix of output layer

protected: 
	FMSNNLearningParamsT m_paramsLearning;	// learning parameters
	FMSNNParamsT m_paramsFMSNN;			// architecture parameters of FMSNN
	
	// assistant variables for training
	double* m_ai;	// input layer
	double* m_af;	// FM layer
	double** m_ahs;	// hidden layers
	double* m_ao;	// output layer

	double* m_df; 	// delta array of FM layer
	double** m_dhs;	// delta arrays of hidden layers
	double* m_do;	// delta array of output layer

//	Matrix m_cf;	// change matrix of weight in FM layer ((ni+1) * nfm)
	Matrix* m_cvfs;	// change matrix of interaction in FM layer (nfm * ni * fm_k)
	Matrix* m_chs;	// change matrices of hidden layers
	Matrix m_co;	// change matrix of output layer
}; 

}

#endif /* _METIS_NN_FMSNN_H */


