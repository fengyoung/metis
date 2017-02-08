// MLP.h
//
// MultiLayer Perceptron (MLP) neural network consists one input layer, one output layer and some hidden layers.
//
// In the MLP architecture, the number of input nodes equls the number of input signal add 1, the "1" for bias.
// The number of output nodes equls the number of output signal add 1, the "1" for bias. 
// The number of hidden layers and each layer node number are user defined, they also add bias automatically.
// 
// IMPORTANT
// (1) The activation function of hidden must be sigmoid, thanh or relu, and the activation fucntion of output 
//     is linear(for regresson), sigmoid(for bi-classification) or softmax(multi-classification).
// (2) relu is recommended activation for hidden layers.
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v3.0 2016-10-24 by fengyoung
//	v2.0 2016-10-06 by fengyoung
//	v1.0 2016-03-14
//

#ifndef _METIS_NEURAL_NETWORK_MLP_H 
#define _METIS_NEURAL_NETWORK_MLP_H 

#include "NeuralNetwork.h"


namespace metis_nn
{


class MLP : public NeuralNetwork
{
public: 
	MLP(); 
	virtual ~MLP(); 
	
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
	
	bool ModelUpdate(NNAssi** ppAssi, const double dLearningRate); 
	
protected:
	bool Create(); 
	void Release(); 

	bool FeedForward(NNAssi** ppAssi, const double* x, const int32_t x_len); 
	bool LayerActivation(double* ao, const double* ai, Matrix& w, const EActType eActType);
	bool BackPropagate(double& dLoss, NNAssi** ppAssi, const double* x, const int32_t x_len, const double* y, const int32_t y_len); 
	bool LayerDeltaBack(double* low_do, const double* up_do, Matrix& w, const bool bOneCol = false); 
	
	bool ModelUpdate_SGD(NNAssi** ppAssi, const double dLearningRate, const ERegula eRegula, const double dLambda);
	bool ModelUpdate_Momentum(NNAssi** ppAssi, const double dLearningRate, const double dBeta);  
	bool ModelUpdate_NAG(NNAssi** ppAssi, const double dLearningRate, const double dBeta); 
	bool ModelUpdate_AdaGrad(NNAssi** ppAssi, const double dLearningRate, const double dEps); 
	bool ModelUpdate_RMSprop(NNAssi** ppAssi, const double dLearningRate, const double dBeta, const double dEps);
	bool ModelUpdate_AdaDelta(NNAssi** ppAssi, const double dRho, const double dBeta, const double dEps); 
	bool ModelUpdate_Adam(NNAssi** ppAssi, const double dLearningRate, const double dBeta1, const double dBeta2, const double dEps); 

public:
	Matrix* m_ws;	// weight matrices  
}; 

}


#endif /* _METIS_NEURAL_NETWORK_MLP_H */ 

 
