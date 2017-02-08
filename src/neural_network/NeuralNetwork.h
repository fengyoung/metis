// NeuralNetwork.h
//
// Base class of NN, which defines some interfaces, include:
// (1) Init & InitFromConfig
// (2) Train & Epoch
// (3) Predict
// (4) Save & Load
// (5) CreateAssi & ReleaseAssi
// (6) Validation & Validation_Binary
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v3.0 2016-10-24 by fengyoung
//	v2.0 2016-10-06 by fengyoung
//	v1.0 2016-03-14
//

#ifndef _METIS_NEURAL_NETWORK_NN_H 
#define _METIS_NEURAL_NETWORK_NN_H 

#include <vector>
using namespace std;
#include "Params.h"
#include "Activation.h"
#include "Pattern.h"
#include "Matrix.h"
#include "NNAssi.h"
#include <stdint.h>


namespace metis_nn
{

// return value
#define _METIS_NN_SUCCESS			0	// success
#define _METIS_NN_ERROR_INPUT_NULL		-1	// the input parameters is null
#define _METIS_NN_ERROR_WRONG_LEN		-2	// length error
#define _METIS_NN_ERROR_MODEL_NULL		-3	// the model is null
#define _METIS_NN_ERROR_FILE_OPEN		-4	// failed to open file
#define _METIS_NN_ERROR_NOT_MODEL_FILE		-5	// is not model file
#define _METIS_NN_ERROR_WEIGHT_MISALIGNMENT	-6	// weight is not aligned
#define _METIS_NN_ERROR_LAYERS_MISMATCHING	-7	// depth is not matched
#define _METIS_NN_ERROR_LERANING_PARAMS		-8	// incorrect learning parameters
#define _METIS_NN_ERROR_ACH_PARAMS		-9	// incorrect architecture parameters 
#define _METIS_NN_ERROR_MODEL_DATA		-10	// incorrect model data (or format)
#define _METIS_NN_ASSI_ERROR			-11	// assistant error


enum ENNType
{
	_NN_PERCEPTRON,
	_NN_MLP
};


// CLASS
// 	NeuralNetwork - the base class of NN, which defines interfaces
class NeuralNetwork
{
public:
	// Construction & Destruction
	NeuralNetwork(const ENNType eNNType);
	virtual ~NeuralNetwork(); 

	// NAME
	// 	Init - initialize the NN
	// 	InitFromConfig - initialize the NN from configure file
	// 
	// DESCRIPTION
	// 	pLearnParams: learning parameters 
	// 	pArchParams: architecture parameters 
	// 	sConfigFile: configure file
	// 	nInput: input size
	// 	nOutput: output size
	//
	// RETURN
	// 	true for success, false for failed 
	virtual bool Init(LearnParams* pLearnParams, ArchParams* pArchParams) = 0; 
	virtual bool InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput) = 0;

	// NAME
	// 	Train - train current NN by some patterns 
	// 	Epoch - one training round that all patterns are scaned
	//
	// DESCRIPTION
	// 	vtrPatts: patterns for training
	// 	dAvgLoss: average loss in epoch
	// 	dLearningRate: learning rate
	// 	nStartOff: start offset of patterns array
	//
	// RETRUN
	// 	true for success, false for failed   
	virtual bool Train(vector<Pattern*>& vtrPatts); 
	virtual bool Epoch(double& dAvgLoss, NNAssi** ppAssi, vector<Pattern*>& vtrPatts, const double dLearningRate, const int32_t nStartOff = 0) = 0; 
	
	// NAME
	// 	Predict - predict by currect NN
	//
	// DESCRIPTION
	// 	y: output predicted values
	// 	y_len: size of y 
	// 	x: input feature vector
	// 	x_len: size of x
	//
	// RETRUN
	//	0 for success, otherwise retrun error code(<0) 	
	virtual int32_t Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len) = 0; 

	// NAME
	// 	Save - save current NN model to file
	// 	Load - load from file to construct current NN model
	//
	// DESCRIPTION
	// 	sModelFile: model file
	//
	// RETURN
	// 	0 for success, otherwise retrun error code(<0)
	virtual int32_t Save(const char* sModelFile) = 0; 
	virtual int32_t Load(const char* sModelFile) = 0;
 
	virtual bool CombineWith(NeuralNetwork* pNN, const double w0 = 1.0, const double w1 = 1.0) = 0;
	virtual void NumericMultiWith(const double a) = 0; 
	virtual bool SetByModelString(const char* sModelStr) = 0; 
	virtual string ConvToModelString() = 0; 
	virtual bool CompArchWith(NeuralNetwork* pNN) = 0; 

	// NAME
	// 	IsNull - detect if the current model is null
	virtual bool IsNull() = 0; 

	// NAME
	// 	CreateAssi - create assistants for prediction or training
	// 	ReleaseAssi - release assistants
	//
	// DESCRIPTION
	// 	bForTrain - if the assistant is created for training
	// 	ppAssi - assistatns
	//
	// RETURN
	// 	assistants
	// 	true for success, false for fail
	virtual NNAssi** CreateAssi(const bool bForTrain) = 0;
	virtual bool ReleaseAssi(NNAssi** ppAssi) = 0; 

	// NAME
	// 	GetNNType - get the NN type
	// 	GetLearnParams - get learning parameters	
	// 	GetArchParams - get architecture parameters	
	// 	SetCanelFlag - set the cancel flag as true or false
	// 	PrintParams - print learning parameters and architecture parameters
	ENNType GetNNType(); 
	LearnParams* GetLearnParams() const; 	
	ArchParams* GetArchParams() const; 	
	bool SetCancelFlag(const bool bCancel); 
	void PrintParams(ostream& os); 
	
	// get the activation type of output layer 
	static EActType OutputActType(const int32_t nOutput); 
	
	// initialize the weight matrix 
	static void WeightMatrixInit(Matrix& w);

	// NAME
	// 	Validation - validate the current NN 
	// 	Validation - validate the current NN, if it's for bi-classification
	double Validation(vector<Pattern*>& vtrValidation, const int32_t nMaxCnt = -1);
	int32_t Validation_Binary(double& dValidatedError, double& dAuc, vector<Pattern*>& vtrValidation, const int32_t nMaxCnt = -1);	

protected:
	ENNType m_eNNType;  
	LearnParams* m_pLearnParams;	// learning parameters
	ArchParams* m_pArchParams;	// architecture parameters
	
	static bool m_bUpdateCancel; 
}; 

}

#endif /* _METIS_NEURAL_NETWORK_NN_H */
 
