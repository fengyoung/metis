#ifndef _METRIX_PREDICTION_MODEL_MLP_H
#define _METRIX_PREDICTION_MODEL_MLP_H

#include "Matrix.h"
#include "TypeDefs.h"
using namespace metis_nn; 
#include "Model.h"


namespace metis_pred
{


class Model_MLP : public Model
{
public: 
	Model_MLP(); 
	virtual ~Model_MLP(); 

	virtual bool Load(const char* sModelFile); 
	virtual bool Save(const char* sModelFile); 
	virtual string ToString(); 
	virtual bool FromString(const char* sStr); 
	virtual void Release(); 

	virtual double Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget = 0); 

	virtual bool CombineWith(Model* pModel, const double w0 = 1.0, const double w1 = 1.0); 

protected:
	void ActivateHiddenLowest(double* up_a, const int32_t up_size, vector<pair<int32_t,double> >& vtrFeat, 
		const EActType up_act_type, Matrix* p_w, Matrix* p_bn = NULL);
	void ActivateHidden(double* up_a, const int32_t up_size, const double* low_a, const int32_t low_size, 
		const EActType up_act_type, Matrix* p_w, Matrix* p_bn = NULL);
	double ActivateOutput(const double* low_a, const int32_t low_size, const EActType up_act_type, 
			Matrix* p_w, Matrix* p_bn, const int32_t nTarget = 0);

public:
	MLPParamsT m_paramsMLP;         // architecture parameters of MLP

	Matrix* m_whs;		// transform matrices of hidden layers
	Matrix* m_bnhs;		// batch normalization matrics of hidden layers (n * 2)

	Matrix m_wo;        // transform matrix
	Matrix m_bno;        // batch normalization matrix of output layer
};

}

#endif /* _METRIX_PREDICTION_MODEL_MLP_H */ 


