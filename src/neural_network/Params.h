#ifndef _METIS_NEURAL_NETWORK_PARAMS_H 
#define _METIS_NEURAL_NETWORK_PARAMS_H 

#include <string>
#include <iostream>
#include <vector>
using namespace std; 
#include <stdint.h>


namespace metis_nn
{

/////////////////////////////////////////////////////////////////////////
// Activation types

enum EActType
{
	_ACT_LINEAR,
	_ACT_SIGMOID,
	_ACT_TANH,
	_ACT_RELU,
	_ACT_RELU6,
	_ACT_SOFTMAX,
	_ACT_NONE
};


class ActConv
{
private: 
	ActConv(); 
	virtual ~ActConv(); 
public: 
	// Tansform activation type to name string 
	static string ActName(const EActType eActType);
	// Retansform activation name string to type 
	static EActType ActType(const char* sActTypeName);
}; 


/////////////////////////////////////////////////////////////////////////
// Regularization types

enum ERegula
{
	_REGULA_L1,     // L1 
	_REGULA_L2,     // L2
	_REGULA_NONE    // no regularization
};


class RegulaConv
{
private: 
	RegulaConv(); 
	virtual ~RegulaConv(); 

public: 
	// Transform regularization type to name string
	static string RegulaName(const ERegula eRegula);
	// Retransform regularization name string to type
	static ERegula RegulaType(const char* sRegulaName);
}; 


/////////////////////////////////////////////////////////////////////////
// Optimization types

enum EOptim
{
	_OPTIM_SGD,		// Standard Stochastic Gradient Descent (SGD) with Mini-Batch 
	_OPTIM_MOMENTUM,	// Momentum method
	_OPTIM_NAG,		// Nesterov Accelerated Gradient algorithm (NAG)
	_OPTIM_ADAGRAD,		// Adaptive Gradinet algorithm (AdaGrad)
	_OPTIM_RMSPROP,		// Root Mean Square Propagation (RMSprop)
	_OPTIM_ADADELTA,	// Adaptive Delta method (AdaDelta)
	_OPTIM_ADAM		// Adaptive Moment Estimation (Adam)
}; 


class OptimConv
{
private: 
	OptimConv(); 
	virtual ~OptimConv(); 
public: 
	// Transform optimization type to name string
	static string OptimName(const EOptim eOptim); 
	// Retransform optimization name string to type
	static EOptim OptimType(const char* sOptimName); 
};


/////////////////////////////////////////////////////////////////////////
// Parameters for each optimization method

class OptimParams
{
public: 
	OptimParams();
	virtual ~OptimParams();

	virtual void Print(ostream& os) = 0;
	virtual bool Read(istream& is) = 0;
	virtual string ToString() = 0;
	virtual bool FromString(const char* sStr) = 0;
	virtual bool FromConfig(const char* sConfigFile) = 0;

	static OptimParams* New(const EOptim eOptim);

public: 
	double learning_rate_init;
}; 


class SGDParams : public OptimParams
{
public: 
	SGDParams();
	SGDParams(const SGDParams& params);
	virtual ~SGDParams();
	
	SGDParams& operator = (const SGDParams& params);

	virtual void Print(ostream& os);
	virtual bool Read(istream& is);
	virtual string ToString();
	virtual bool FromString(const char* sStr);
	virtual bool FromConfig(const char* sConfigFile); 

public: 
	ERegula regula; 
	double lambda;
}; 


class MomentumParams: public OptimParams
{
public: 
	MomentumParams();
	MomentumParams(const MomentumParams& params);
	virtual ~MomentumParams();
	
	MomentumParams& operator = (const MomentumParams& params);

	virtual void Print(ostream& os);
	virtual bool Read(istream& is);
	virtual string ToString();
	virtual bool FromString(const char* sStr);
	virtual bool FromConfig(const char* sConfigFile); 

public: 
	double beta; 
};


class NAGParams: public OptimParams
{
public: 
	NAGParams();
	NAGParams(const NAGParams& params);
	virtual ~NAGParams();
	
	NAGParams& operator = (const NAGParams& params);

	virtual void Print(ostream& os);
	virtual bool Read(istream& is);
	virtual string ToString();
	virtual bool FromString(const char* sStr);
	virtual bool FromConfig(const char* sConfigFile); 

public: 
	double beta; 
};


class AdaGradParams: public OptimParams
{
public: 
	AdaGradParams();
	AdaGradParams(const AdaGradParams& params);
	virtual ~AdaGradParams();
	
	AdaGradParams& operator = (const AdaGradParams& params);

	virtual void Print(ostream& os);
	virtual bool Read(istream& is);
	virtual string ToString();
	virtual bool FromString(const char* sStr);
	virtual bool FromConfig(const char* sConfigFile); 

public: 
	double eps;
};


class RMSpropParams: public OptimParams
{
public: 
	RMSpropParams();
	RMSpropParams(const RMSpropParams& params);
	virtual ~RMSpropParams();
	
	RMSpropParams& operator = (const RMSpropParams& params);

	virtual void Print(ostream& os);
	virtual bool Read(istream& is);
	virtual string ToString();
	virtual bool FromString(const char* sStr);
	virtual bool FromConfig(const char* sConfigFile); 

public: 
	double beta;
	double eps;
};


class AdaDeltaParams: public OptimParams
{
public: 
	AdaDeltaParams();
	AdaDeltaParams(const AdaDeltaParams& params);
	virtual ~AdaDeltaParams();
	
	AdaDeltaParams& operator = (const AdaDeltaParams& params);

	virtual void Print(ostream& os);
	virtual bool Read(istream& is);
	virtual string ToString();
	virtual bool FromString(const char* sStr);
	virtual bool FromConfig(const char* sConfigFile); 

public: 
	double rho;
	double beta;
	double eps;
};


class AdamParams: public OptimParams
{
public: 
	AdamParams();
	AdamParams(const AdamParams& params);
	virtual ~AdamParams();
	
	AdamParams& operator = (const AdamParams& params);

	virtual void Print(ostream& os);
	virtual bool Read(istream& is);
	virtual string ToString();
	virtual bool FromString(const char* sStr);
	virtual bool FromConfig(const char* sConfigFile); 

public: 
	double beta1;
	double beta2;
	double eps;
};


/////////////////////////////////////////////////////////////////////////
// Learning Parameters

class LearnParams
{
public:
	LearnParams(); 
	LearnParams(const LearnParams& learnParams); 
	virtual ~LearnParams(); 
	
	LearnParams& operator = (const LearnParams& learnParams); 
	
	virtual void Print(ostream& os);
        virtual bool Read(istream& is);
        virtual string ToString();
        virtual bool FromString(const char* sStr);
	virtual bool FromConfig(const char* sConfigFile); 

public:
	EOptim optim;		// method of optimization, default 'NAG'
	int32_t batch_size;	// batch size of iteration, default 200	
	int32_t max_epoches;	// maximal number of epoches, default 200
	int32_t early_stop;	// epoches for early stop, 0 denotes disable, default 10
        double epsilon;		// threshold of loss, for iteration stopping, default 0.1
	OptimParams* p_optim_params; 	// optimization parameters 
}; 


/////////////////////////////////////////////////////////////////////////
// Architecture Parameters

enum EArchParamsType
{
	_ARCH_PARAMS_BASIC,
	_ARCH_PARAMS_MLP
}; 


class ArchParams
{
public: 
	ArchParams(); 
	ArchParams(const ArchParams& archParams); 
	virtual ~ArchParams(); 
	
	ArchParams& operator = (const ArchParams& archParams); 

	virtual void Print(ostream& os);
        virtual bool Read(istream& is);
        virtual string ToString();
        virtual bool FromString(const char* sStr);		
	virtual bool FromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput); 

	EArchParamsType GetType(); 

protected: 
	EArchParamsType type; 
 	
public: 
	int32_t input;	// number of input 
	int32_t output;	// number of output 
};


class ArchParams_MLP : public ArchParams
{
public:
	ArchParams_MLP(); 
	ArchParams_MLP(const ArchParams_MLP& archParamsMLP); 
	virtual ~ArchParams_MLP(); 
	
	ArchParams_MLP& operator = (const ArchParams_MLP& archParamsMLP); 

	virtual void Print(ostream& os);
        virtual bool Read(istream& is);
        virtual string ToString();
        virtual bool FromString(const char* sStr);
	virtual bool FromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput); 

public:
	EActType hidden_act;		// activation type of hidden layers
	vector<int32_t> vtr_hiddens;	// number of each hidden layer
};


}


#endif /* _METIS_NEURAL_NETWORK_PARAMS_H */
 
