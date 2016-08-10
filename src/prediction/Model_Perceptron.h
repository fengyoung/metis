#ifndef _METIS_PREDICTION_MODEL_PERCEPTRON_H
#define _METIS_PREDICTION_MODEL_PERCEPTRON_H

#include "Matrix.h"
#include "TypeDefs.h"
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
	virtual void Release(); 
	
	virtual double Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget = 0); 

	virtual bool CombineWith(Model* pModel, const double w0 = 1.0, const double w1 = 1.0); 

public: 
	PerceptronParamsT m_paramsPerceptron;   // architecture parameters of perceptron
	Matrix m_wo;        // transform matrix
};

}

#endif /* _METIS_PREDICTION_MODEL_PERCEPTRON_H */ 


