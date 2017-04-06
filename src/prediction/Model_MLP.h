#ifndef _METIS_PREDICTION_MODEL_MLP_H
#define _METIS_PREDICTION_MODEL_MLP_H

#include "Matrix.h"
#include "MLP.h"
using namespace metis_nn; 
#include "Model.h"


namespace metis_pred 
{


class Model_MLP: public Model
{
public: 
	Model_MLP(); 
	virtual ~Model_MLP(); 

	virtual bool Load(const char* sModelFile); 
	virtual bool Save(const char* sModelFile); 
	virtual string ToString(); 
	virtual bool FromString(const char* sStr); 
	
	virtual double Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget = 0, const bool bOutliersCheck = true); 
	virtual double Predict(const double* x, const int32_t nLen, const int32_t nTarget = 0); 

	virtual int32_t N_Input(); 
	virtual int32_t N_Output(); 

private: 	
	MLP m_mlp; 
};

}

#endif /* _METIS_PREDICTION_MODEL_MLP_H*/ 


