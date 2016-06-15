// Perceptron.h
//
// Single preceptron only consists one input layer and one output layer.
//
// In the perceptron architecture, the number of input nodes equls the number of input signal add 1, the "1" for bias.
// The number of output nodes equls the number of output signal 
// Nodes on output layer take activation functions   
//
// IMPORTANT
// (1) when using the logit (sigmoid) activation function for the output layer make sure y values are scaled from 0 to 1
// (2) when using the tanh activation function for the output layer make sure y values are scaled from 0 to 1
// (3) sigmoid function is suggested in output layer for binary classification, and softmax is suggested for multiple classfication
// (4) the output activation can be set as 'none' for regression
//
// AUTHOR
//	fengyoung (fengyoung82@sina.com)
// 
// HISTORY
//	v1.0 2016-03-14
//

#ifndef _METIS_NN_PERCEPTRON_H 
#define _METIS_NN_PERCEPTRON_H 

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
//	Perceptron - single perceptron 
//
// DESCRIPTION
//	This perceptron is based on Back-Propagation (BP) algorithm. 
//	The training supports online(SGD), mini-batch(MSGD) and batch(GD) mode
//
class Perceptron
{
public: 
	Perceptron();
	virtual ~Perceptron();

	// NAME
	//	Init - initialize the perceptron parameters, including learning parameters and architecture parameters 
	//	InitFromConfig - initialize the perceptron parameters by values read from config file
	// 
	// DESCRIPTION
	//	perceptronParamsT: architecture parameters of perceptron
	//	perceptronLearningParamsT: learning parameters
	//	sConfigFile: config file
	//	nInput: number of input signal 
	//	nOnput: number of output signal 
	//
	// RETURN
	//	true for success, false for some errors
	void Init(const PerceptronParamsT perceptronParamsT, const PerceptronLearningParamsT perceptronLearningParamsT); 
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
	//	Save - save current perceptron model to file
	//	Load - load perceptron model from file to construct current object
	//
	// DESCRIPTION
	//	sFile - perceptron model file
	// 
	// RETURN
	//	Return _FYDL_SUCCESS for success, others for some errors
	int32_t Save(const char* sFile); 
	int32_t Load(const char* sFile); 

	// Get learning parameters
	PerceptronLearningParamsT GetLearningParams(); 
	
	// Get architecture parameters 
	PerceptronParamsT GetArchParams(); 

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
	//	ModelUpdate - update transform matrix in perceptron
	// 
	// DESCRIPTION
	//	After transform matrices updating, elements of all change matrices should be set as 0
	//	
	//	learning_rate: learning rate
	//	eRegula: regularzation type
	//	nPattCnt: count of training patterns	
	void ModelUpdate(const double learning_rate, const ERegula eRegula, const int32_t nPattCnt);
	
	// NAME
	//	Validation - validate current perceptron model
	// 
	// DESCRIPTION	
	//	vtrPatts: training patterns list
	//	nBackCnt: number of validation patterns, which are at back of the list
	// 
	// RETRUN
	//	The precision and RMSE
	pair<double, double> Validation(vector<Pattern*>& vtrPatts, const int32_t nBackCnt);


	////////////////////////////////////////////////////////////////////////////

public: 	
	bool SetByModelString(const char* sStr); 
	string ConvToModelString(); 	
	bool CombineWith(Perceptron& perc, const double w0, const double w1);


	////////////////////////////////////////////////////////////////////////////
public: 
	// model variables
	Matrix m_wo;		// transform matrix

protected: 
	PerceptronLearningParamsT m_paramsLearning;	// learning parameters
	PerceptronParamsT m_paramsPerceptron;	// architecture parameters of perceptron

	// assistant variables for training
	double* m_ai;	// input layer
	double* m_ao;	// output layer
	double* m_do;	// delta array of output layer
	Matrix m_co;	// change matrix of output layer
}; 

}

#endif /* _METIS_NN_PERCEPTRON_H */

