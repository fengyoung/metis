#include "Model.h"
#include "Model_Perceptron.h"
#include "Model_FM.h"
#include "Model_MLP.h"
#include "Model_FMSNN.h"
using namespace metis_pred; 


Model::Model(const ModelType modelType) : m_modelType(modelType)
{
}


Model::~Model()
{
}


ModelType Model::GetType()
{
	return m_modelType;
}


Model* Model::LoadModel(const char* sModelFile)
{
	if(!sModelFile)
		return NULL; 
	
	Model* p_model = new Model_Perceptron();
	if(p_model->Load(sModelFile))
		return p_model; 
	
	delete p_model;	
	p_model = new Model_FM(); 
	if(p_model->Load(sModelFile))
		return p_model; 
	
	delete p_model;	
	p_model = new Model_MLP(); 
	if(p_model->Load(sModelFile))
		return p_model; 
	
	delete p_model;	
	p_model = new Model_FMSNN(); 
	if(p_model->Load(sModelFile))
		return p_model; 

	delete p_model; 
	return NULL; 
}


bool Model::SaveModel(const char* sModelFile, Model* pModel)
{
	if(!sModelFile || !pModel)
		return false; 
	return pModel->Save(sModelFile); 
}


string Model::ConvToString(Model* pModel)
{
	if(!pModel)
		return string(""); 
	return pModel->ToString(); 	
}


Model* Model::ParseModelFromString(const char* sStr)
{
	if(!sStr)
		return NULL; 
	
	Model* p_model = new Model_Perceptron();
	if(p_model->FromString(sStr))
		return p_model; 
	
	delete p_model;	
	p_model = new Model_MLP(); 
	if(p_model->FromString(sStr))
		return p_model; 
	
	delete p_model; 
	return NULL; 
}




