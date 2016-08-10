// FM.h
//
// Factorization Machine like Perceptron only consists one input layer and one output layer.
// fm(x) = w0 + w1 * x1 + w2 * x2 + ... + SUMi{SUMj[<vi, vj> * xi * xj]}
//
// Contrast to Perceptron, there are one transform matrix and M (equals to output number) interaction matrices between input and output layer 
// Nodes on output layer take activation functions   
//
// IMPORTANT
// (1) FM is more suitable for onehot and sparse samples
// (2) when using the tanh activation function for the output layer make sure y values are scaled from 0 to 1
// (3) sigmoid function is suggested in output layer for binary classification, and softmax is suggested for multiple classfication
// (4) the output activation can be set as 'none' for regression
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v1.0 2016-07-30
//

#ifndef _METIS_NN_FM_H 
#define _METIS_NN_FM_H 


#include <string>
#include <vector>
using namespace std;
#include <stdint.h>
#include "TypeDefs.h"
#include "Pattern.h"
#include "Matrix.h"



namespace metis_nn
{


// CLASS
//	FM - Factorization Machine 
//
// DESCRIPTION
//	This FM is based on Back-Propagation (BP) algorithm. 
//	The training supports online(SGD), mini-batch(MSGD) and batch(GD) mode
//
class FM
{
public: 
	// Construction & Destruction
	FM();
	virtual ~FM();

	// NAME
	//	Init - initialize the FM parameters, including learning parameters and architecture parameters 
	//	InitFromConfig - initialize the FM parameters by values read from config file
	// 
	// DESCRIPTION
	//	fmParamsT: architecture parameters of FM
	//	fmLearningParamsT: learning parameters
	//	sConfigFile: config file
	//	nInput: number of input signal 
	//	nOnput: number of output signal 
	//
	// RETURN
	//	true for success, false for some errors
	void Init(const FMParamsT fmParamsT, const FMLearningParamsT fmLearningParamsT); 
	bool InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput);
	
	// NAME
	//	Train - train perceptron model by patterns
	//	TrainAndValidate - train perceptron model with validation	
	// 
	// DESCRIPTION
	//	vtrPatts: list of training patterns
	void Train(vector<Pattern*>& vtrPatts);
	void TrainAndValidate(vector<Pattern*>& vtrPatts);	

	// NAME
	//	Predict - calculate prediction of input signal based on current perceptron model
	// 
	// DESCRIPTION
	//	y: output parameter, the prediction result
	//	y_len: size of y
	//	x: input signal
	//	x_len: size of x
	// 
	// RETURN
	//	Return _PERCEPTRON_SUCCESS for success, others for some errors
	int32_t Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len); 
	
	// NAME
	//	Save - save current FM model to file
	//	Load - load FM model from file to construct current object
	//
	// DESCRIPTION
	//	sFile - FM model file
	// 
	// RETURN
	//	Return _FYDL_SUCCESS for success, others for some errors
	int32_t Save(const char* sFile); 
	int32_t Load(const char* sFile); 

	// Get learning parameters
	FMLearningParamsT GetLearningParams(); 
	
	// Get architecture parameters 
	FMParamsT GetArchParams(); 

	// Set current FM by model string
	bool SetByModelString(const char* sModelStr); 
	// Convert current FM to model string
	string ConvToModelString(); 	
	// Combine current FM which another model
	bool CombineWith(FM& fm, const double w0, const double w1);

protected:
	// Create inner objects, allocate menory
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
	//	ModelUpdate - update transform matrix in FM
	// 
	// DESCRIPTION
	//	After transform matrices updating, elements of all change matrices should be set as 0
	//	
	//	learning_rate: learning rate
	//	eRegula: regularzation type
	//	nPattCnt: count of training patterns	
	void ModelUpdate(const double learning_rate, const ERegula eRegula, const int32_t nPattCnt);
	
	// NAME
	//	Validation - validate current FM model
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
	Matrix m_wo;		// weight matrix ((ni+1) * no)
	Matrix* m_vos;		// interaction matrices (no * ni * k)

protected: 
	FMLearningParamsT m_paramsLearning;	// learning parameters
	FMParamsT m_paramsFM;	// architecture parameters of FM 

	// assistant variables for training
	double* m_ai;	// input layer
	double* m_ao;	// output layer
	double* m_do;	// delta array of output layer
	Matrix m_co;	// change matrix of output layer for weight 
	Matrix* m_cvos;	// change matrices of output layer for interaction (no * ni * k)
}; 

}


#endif /* _METIS_NN_FM_H */ 


