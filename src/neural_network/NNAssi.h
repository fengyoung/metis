#ifndef _METIS_NEURAL_NETWORK_NN_ASSISTANT_H 
#define _METIS_NEURAL_NETWORK_NN_ASSISTANT_H 

#include "Matrix.h"
#include "Params.h"
#include <stdint.h>


namespace metis_nn
{


// CLASS
//	NNAssi - assistant variables for layer-training 
class NNAssi
{
public:
	NNAssi(const EOptim eOptim);
	virtual ~NNAssi(); 

	// NAME
	// 	Create - create inner parameters of current assistant
	// 	Release - relese inner parameters of current assistant
	//
	// DESCRIPTION
	// 	nInput - size of input 
	// 	nOutput - size of output
	// 	bForTrain - if the assistant is created for training
	virtual void Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain = true) = 0; 
	virtual void Release() = 0; 

	// NAME
	// 	New - constract a new assistant object 
	//
	// DESCRIPTION
	// 	eOptim - type of optimization
	// 	nInput - size of input 
	// 	nOutput - size of output
	// 	bForTrain - if the assistant is created for training
	// 
	// RETURN
	// 	assistant object
	static NNAssi* New(const EOptim eOptim, const int32_t nInput, const int32_t nOutput, const bool bForTrain = true); 

public:
	EOptim m_eOptim; 	// type of optimization 
	double* m_ao;		// nOutput layer
	double* m_do;	// delta array of nOutput layer
	Matrix m_g;	// gradient matrix of loss function 
};


// CLASS
// 	NNAssi_SGD - assistant for SGD
class NNAssi_SGD : public NNAssi
{
public: 
	NNAssi_SGD(); 
	virtual ~NNAssi_SGD(); 
	
	virtual void Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain = true); 
	virtual void Release(); 

}; 


// CLASS
// 	NNAssi_Momentum - assistant for momentum
class NNAssi_Momentum : public NNAssi
{
public: 
	NNAssi_Momentum(); 
	virtual ~NNAssi_Momentum(); 
	
	virtual void Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain = true); 
	virtual void Release(); 

public: 
	Matrix m_v_prev;// previous descent direction
}; 


// CLASS
// 	NNAssi_NAG - assistant for NAG
class NNAssi_NAG : public NNAssi
{
public: 
	NNAssi_NAG(); 
	virtual ~NNAssi_NAG(); 
	
	virtual void Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain = true); 
	virtual void Release(); 

public: 
	Matrix m_v_prev;	// previous descent direction
}; 


// CLASS
// 	NNAssi_AdaGrad - assistant for AdaGrad 
class NNAssi_AdaGrad : public NNAssi
{
public: 
	NNAssi_AdaGrad(); 
	virtual ~NNAssi_AdaGrad(); 
	
	virtual void Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain = true); 
	virtual void Release(); 

public: 
	Matrix m_g2_acc;	// grad square accumulated
}; 


// CLASS
// 	NNAssi_RMSProp - assistant for RMSProp
class NNAssi_RMSprop : public NNAssi
{
public: 
	NNAssi_RMSprop(); 
	virtual ~NNAssi_RMSprop(); 
	
	virtual void Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain = true); 
	virtual void Release(); 

public: 
	Matrix m_g2_mavg;		// moving average of squared gradients
}; 


// CLASS
// 	NNAssi_AdaDelta - assistant for AdaDelta 
class NNAssi_AdaDelta : public NNAssi
{
public: 
	NNAssi_AdaDelta(); 
	virtual ~NNAssi_AdaDelta(); 
	
	virtual void Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain = true); 
	virtual void Release(); 

public: 
	Matrix m_v2_mavg;	// moving average of squared descent 
	Matrix m_g2_mavg;	// moving average of squared gradients
}; 


// CLASS
// 	NNAssi_Adam - assistant for Adam
class NNAssi_Adam : public NNAssi
{
public: 
	NNAssi_Adam(); 
	virtual ~NNAssi_Adam(); 
	
	virtual void Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain = true); 
	virtual void Release(); 

public: 
	Matrix m_g_mavg;	// moving average of gradients 
	Matrix m_g2_mavg;	// moving average of squared gradients
}; 

}


#endif /* _METIS_NEURAL_NETWORK_NN_ASSISTANT_H */

