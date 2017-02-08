// Model.h
//
// Base class of NN Model for prediction, which defines some interfaces, include:
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v3.0 2016-10-24 by fengyoung
//	v2.0 2016-10-06 by fengyoung
//	v1.0 2016-03-14
//

#ifndef _METIS_PREDICTION_MODEL_H
#define _METIS_PREDICTION_MODEL_H

#include <string>
#include <vector>
#include <iostream>
using namespace std; 
#include <stdint.h>


namespace metis_pred
{

enum ModelType
{
	_MODEL_PERCEPTRON,
	_MODEL_MLP
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

	virtual double Predict(vector<pair<int32_t,double> >& vtrFeat, const int32_t nTarget = 0) = 0;
	virtual double Predict(const double* x, const int32_t nLen, const int32_t nTarget = 0) = 0; 

	virtual int32_t N_Input() = 0; 
	virtual int32_t N_Output() = 0; 

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


