#ifndef _METIS_PREDICTION_MODEL_FMSNN_H 
#define _METIS_PREDICTION_MODEL_FMSNN_H 

#include "Matrix.h"
#include "TypeDefs.h"
using namespace metis_nn; 
#include "Model.h"


namespace metis_pred 
{


class Model_FMSNN : public Model
{
public: 
	Model_FMSNN(); 
	virtual ~Model_FMSNN(); 

	virtual bool Load(const char* sModelFile); 
	virtual bool Save(const char* sModelFile); 
	virtual string ToString(); 
	virtual bool FromString(const char* sModelStr); 
	virtual void Release(); 
	
	virtual double Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget = 0); 

	virtual bool CombineWith(Model* pModel, const double w0 = 1.0, const double w1 = 1.0); 

protected:
	void ActivateFMLayer(double* af, const int32_t inter_size, vector<pair<int32_t,double> >& vtrFeat, 
			Matrix* vfs, const int32_t fm_k, const EActType act_type); 
	void ActivateHidden(double* up_a, const int32_t up_size, const double* low_a, const int32_t low_size, 
			Matrix& w, const EActType up_act_type);
	double ActivateOutput(const double* low_a, const int32_t low_size, Matrix& w, const EActType up_act_type, 
			const int32_t nTarget = 0);

public: 
	FMSNNParamsT m_paramsFMSNN;			// architecture parameters of FMSNN
//	Matrix m_wf;	// weight matrix of FM layer ((ni+1) * nfm)
	Matrix* m_vfs;	// interaction matrices of FM layer (nfm * ni * k)
	Matrix* m_whs;	// transform matrices of hidden layers
	Matrix m_wo;	// transform matrix of output layer
};

}

#endif /* _METIS_PREDICTION_MODEL_FMSNN_H */ 


