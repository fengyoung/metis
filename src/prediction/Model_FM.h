#ifndef _METIS_PREDICTION_MODEL_FM_H 
#define _METIS_PREDICTION_MODEL_FM_H 

#include "Matrix.h"
#include "TypeDefs.h"
using namespace metis_nn; 
#include "Model.h"


namespace metis_pred 
{


class Model_FM : public Model
{
public: 
	Model_FM(); 
	virtual ~Model_FM(); 

	virtual bool Load(const char* sModelFile); 
	virtual bool Save(const char* sModelFile); 
	virtual string ToString(); 
	virtual bool FromString(const char* sStr); 
	virtual void Release(); 
	
	virtual double Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget = 0); 

	virtual bool CombineWith(Model* pModel, const double w0 = 1.0, const double w1 = 1.0); 

public: 
	FMParamsT m_paramsFM;   // architecture parameters of FM 
	Matrix m_wo;        // weight matrix
	Matrix* m_vo;        // interaction matrix
	//Matrix m_bn;        // n * 2 matrix, batch normalization matrix (average & std-deviation)
};

}

#endif /* _METIS_PREDICTION_MODEL_FM_H */ 


