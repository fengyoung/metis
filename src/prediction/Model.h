#ifndef _METIS_PREDICTION_MODEL_H
#define _METIS_PREDICTION_MODEL_H

#include <string>
#include <vector>
using namespace std; 
#include <stdint.h>


namespace metis_pred
{

enum ModelType
{
	_MODEL_PERCEPTRON,
	_MODEL_FM,
	_MODEL_MLP,
	_MODEL_FMSNN
};


class Model
{
public: 
	Model(const ModelType modelType); 
	virtual ~Model(); 

	virtual bool Load(const char* sModelFile) = 0; 
	virtual bool Save(const char* sModelFile) = 0; 
	virtual string ToString() = 0; 
	virtual bool FromString(const char* sStr) = 0; 
	virtual void Release() = 0; 

	virtual double Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget = 0) = 0;

	virtual bool CombineWith(Model* pModel, const double w0 = 1.0, const double w1 = 1.0) = 0; 

	ModelType GetType();

	static Model* LoadModel(const char* sModelFile); 
	static bool SaveModel(const char* sModelFile, Model* pModel); 
	static string ConvToString(Model* pModel); 
	static Model* ParseModelFromString(const char* sStr); 

protected: 
	ModelType m_modelType; 		
};

}

#endif /* _METIS_PREDICTION_MODEL_H */ 


