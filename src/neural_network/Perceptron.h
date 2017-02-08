// Perceptron.h
//
// Single preceptron only consists one input layer and one output layer.
//
// In the perceptron architecture, the number of input nodes equls the number of input signal add 1, the "1" for bias.
// The number of output nodes equls the number of output signal 
// Nodes on output layer take activation functions   
//
// IMPORTANT
// the activation fucntion of output is linear(for regresson), sigmoid(for bi-classification) or softmax(multi-classification).
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v3.0 2016-01-24 by fengyoung
//	v2.0 2016-10-06 by fengyoung
//	v1.0 2016-03-14

#ifndef _METIS_NEURAL_NETWORK_PERCEPTRON_H 
#define _METIS_NEURAL_NETWORK_PERCEPTRON_H 

#include "NeuralNetwork.h"


namespace metis_nn
{


class Perceptron : public NeuralNetwork
{
public: 
	Perceptron(); 
	virtual ~Perceptron(); 
	
	virtual bool Init(LearnParams* pLearnParams, ArchParams* pArchParams); 
	virtual bool InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput); 
	virtual bool Epoch(double& dAvgLoss, NNAssi** ppAssi, vector<Pattern*>& vtrPatts, const double dLearningRate, const int32_t nStartOff = 0); 
	virtual int32_t Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len); 

	virtual int32_t Save(const char* sModelFile); 
	virtual int32_t Load(const char* sModelFile); 
	virtual bool CombineWith(NeuralNetwork* pNN, const double w0 = 1.0, const double w1 = 1.0); 
	virtual void NumericMultiWith(const double a); 
	virtual bool SetByModelString(const char* sModelStr); 
	virtual string ConvToModelString(); 
	virtual bool CompArchWith(NeuralNetwork* pNN); 
	virtual bool IsNull(); 
	
	virtual NNAssi** CreateAssi(const bool bForTrain); 
	virtual bool ReleaseAssi(NNAssi** ppAssi); 

	bool ModelUpdate(NNAssi* pAssi, const double dLearningRate); 

protected:
	bool Create(); 
	void Release(); 
	
	bool FeedForward(NNAssi* pAssi, const double* x, const int32_t x_len); 
	bool BackPropagate(double& dLoss, NNAssi* pAssi, const double* x, const int32_t x_len, const double* y, const int32_t y_len); 

	bool ModelUpdate_SGD(NNAssi* pAssi, const double dLearningRate, const ERegula eRegula, const double dLambda);
	bool ModelUpdate_Momentum(NNAssi* pAssi, const double dLearningRate, const double dBeta);  
	bool ModelUpdate_NAG(NNAssi* pAssi, const double dLearningRate, const double dBeta); 
	bool ModelUpdate_AdaGrad(NNAssi* pAssi, const double dLearningRate, const double dEps); 
	bool ModelUpdate_RMSprop(NNAssi* pAssi, const double dLearningRate, const double dBeta, const double dEps);
	bool ModelUpdate_AdaDelta(NNAssi* pAssi, const double dRho, const double dBeta, const double dEps); 
	bool ModelUpdate_Adam(NNAssi* pAssi, const double dLearningRate, const double dBeta1, const double dBeta2, const double dEps); 

public:
	Matrix m_w; 	// weight matrix
}; 

}


#endif /* _METIS_NEURAL_NETWORK_PERCEPTRON_H */ 

 
