#ifndef _METIS_PREDICTION_MODEL_PERCEPTRON_H
#define _METIS_PREDICTION_MODEL_PERCEPTRON_H

#include "Matrix.h"
#include "Perceptron.h"
using namespace metis_nn; 
#include "Model.h"


namespace metis_pred 
{


class Model_Perceptron : public Model
{
public: 
	Model_Perceptron(); 
	virtual ~Model_Perceptron(); 

	virtual bool Load(const char* sModelFile); 
	virtual bool Save(const char* sModelFile); 
	virtual string ToString(); 
	virtual bool FromString(const char* sStr); 
	
	virtual double Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget = 0); 
	virtual double Predict(const double* x, const int32_t nLen, const int32_t nTarget = 0); 

	virtual int32_t N_Input(); 
	virtual int32_t N_Output(); 

private: 	
	Perceptron m_percep; 
};

}

#endif /* _METIS_PREDICTION_MODEL_PERCEPTRON_H */ 


