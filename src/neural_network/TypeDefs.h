#ifndef _METIS_NN_TYPEDEFS_H 
#define _METIS_NN_TYPEDEFS_H 

#include <iostream>
#include <string>
#include <vector>
using namespace std; 
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
#define _METIS_NN_ERROR_LERANING_PARAMS		-8
#define _METIS_NN_ERROR_ACH_PARAMS		-9
#define _METIS_NN_ERROR_MODEL_DATA		-10


// activation types
enum EActType
{
	_ACT_LINEAR,
	_ACT_SIGMOID,
	_ACT_TANH,
	_ACT_RELU,
	_ACT_SOFTMAX,
	_ACT_NONE
};


// regularization types
enum ERegula
{
	_REGULA_L1,     // L1 
	_REGULA_L2,     // L2
	_REGULA_NONE    // no regularization
};


//////////////////////////////////////////////////////////////////
// typedefs of Perceptron & MLP

// learning parameters of perceptron, FM, MLP & FMSNN
typedef struct _perceptron_learning_params_t
{
	ERegula regula;			// regularization type
	int32_t mini_batch;		// mini_batch; 0 for batch-GD, 1 for SGD, greater than 1 for mini-batch
	int32_t iterations;		// maximal iteration number
	double learning_rate;		// learning rate
	double rate_decay;		// decay of learning rate
	double epsilon;			// threshold of RMSE, used for iteration stopping
	bool batch_norm;		// if need to do batch normalization
} PerceptronLearningParamsT, MLPLearningParamsT, FMLearningParamsT, FMSNNLearningParamsT; 


// architecture parameters of perceptron
typedef struct _perceptron_params_t
{
	int32_t input;			// number of input nodes
	int32_t output;			// number of output nodes
	EActType act_output;	// activation of output layer
} PerceptronParamsT;


// architecture parameters of MLP
typedef struct _mlp_params_t
{
	int32_t input;			// number of input nodes
	int32_t output;			// number of output nodes
	vector<int32_t> vtr_hidden;		// numbers of hidden nodes for each hidden layer
	EActType act_hidden;			// activation of hidden layer
	EActType act_output;			// activation of output layer
} MLPParamsT;


// architecture parameters of FM
typedef struct _fm_params_t
{
	int32_t input;		// number of input nodes
	int32_t output;		// number of output nodes
	int32_t fm_k;		// K value of fm
	EActType act_output;	// activation of output layer
} FMParamsT;


// architecture parameters of FMSNN
typedef struct _fmsnn_params_t
{
	int32_t input;			// number of input nodes
	int32_t output;			// number of output nodes
	int32_t fm_inter;		// number of interaction nodes in fm-layer
	int32_t fm_k;			// K value of fm
	vector<int32_t> vtr_hidden;		// numbers of hidden nodes for each hidden layer
	EActType act_fm_layer;			// activation of FM layer
	EActType act_hidden;			// activation of hidden layer
	EActType act_output;			// activation of output layer
} FMSNNParamsT;




class TypeDefs
{
private:
	TypeDefs(); 
	virtual ~TypeDefs(); 

public: 
	static string ActName(const EActType eActType);
	static EActType ActType(const char* sActTypeName);

	static string RegulaName(const ERegula eRegula); 
	static ERegula RegulaType(const char* sRegulaName);

	static void Print_PerceptronLearningParamsT(ostream& os, const PerceptronLearningParamsT perceptronLearningParamsT); 
	static bool Read_PerceptronLearningParamsT(PerceptronLearningParamsT& perceptronLearningParamsT, istream& is); 
	static string ToString_PerceptronLearningParamsT(const PerceptronLearningParamsT perceptronLearningParamsT); 
	static bool FromString_PerceptronLearningParamsT(PerceptronLearningParamsT& perceptronLearningParamsT, const char* sStr); 

	static void Print_PerceptronParamsT(ostream& os, const PerceptronParamsT perceptronParamsT); 
	static bool Read_PerceptronParamsT(PerceptronParamsT& perceptronParamsT, istream& is); 
	static string ToString_PerceptronParamsT(const PerceptronParamsT perceptronParamsT); 
	static bool FromString_PerceptronParamsT(PerceptronParamsT& perceptronParamsT, const char* sStr); 
	static bool IsEqual_PerceptronParamsT(const PerceptronParamsT perceptronParamsT_1, const PerceptronParamsT perceptronParamsT_2); 

	static void Print_MLPParamsT(ostream& os, const MLPParamsT mlpParamsT); 
	static bool Read_MLPParamsT(MLPParamsT& mlpParamsT, istream& is); 
	static string ToString_MLPParamsT(const MLPParamsT mlpParamsT); 
	static bool FromString_MLPParamsT(MLPParamsT& mlpParamsT, const char* sStr); 
	static bool IsEqual_MLPParamsT(const MLPParamsT mlpParamsT_1, const MLPParamsT mlpParamsT_2); 
	
	static void Print_FMParamsT(ostream& os, const FMParamsT fmParamsT); 
	static bool Read_FMParamsT(FMParamsT& fmParamsT, istream& is); 
	static string ToString_FMParamsT(const FMParamsT fmParamsT); 
	static bool FromString_FMParamsT(FMParamsT& fmParamsT, const char* sStr); 
	static bool IsEqual_FMParamsT(const FMParamsT fmParamsT_1, const FMParamsT fmParamsT_2); 

	static void Print_FMSNNParamsT(ostream& os, const FMSNNParamsT fnnParamsT); 
	static bool Read_FMSNNParamsT(FMSNNParamsT& fnnParamsT, istream& is); 
	static string ToString_FMSNNParamsT(const FMSNNParamsT fnnParamsT); 
	static bool FromString_FMSNNParamsT(FMSNNParamsT& fnnParamsT, const char* sStr); 
	static bool IsEqual_FMSNNParamsT(const FMSNNParamsT fnnParamsT_1, const FMSNNParamsT fnnParamsT_2); 
};

}

#endif /* _METIS_NN_TYPEDEFS_H */ 


