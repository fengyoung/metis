#include "TypeDefs.h"
using namespace metis_nn;
#include "StringArray.h"
using namespace metis_uti;
#include <stdio.h>
#include <string.h>

TypeDefs::TypeDefs()
{
}


TypeDefs::~TypeDefs()
{
}


string TypeDefs::ActName(const EActType eActType)
{
	switch(eActType)
	{
		case _ACT_LINEAR:
			return "linear";
		case _ACT_SIGMOID:
			return "sigmoid";
		case _ACT_TANH:
			return "tanh";
		case _ACT_RELU:
			return "relu";
		case _ACT_SOFTMAX: 
			return "softmax";
		default:
			break;
	}
	return "none";
}


EActType TypeDefs::ActType(const char* sActTypeName)
{
	if(strcmp(sActTypeName, "linear") == 0)
		return _ACT_LINEAR; 
	else if(strcmp(sActTypeName, "sigmoid") == 0)
		return _ACT_SIGMOID; 
	else if(strcmp(sActTypeName, "tanh") == 0)
		return _ACT_TANH; 
	else if(strcmp(sActTypeName, "relu") == 0)
		return _ACT_RELU; 
	else if(strcmp(sActTypeName, "softmax") == 0)
		return _ACT_SOFTMAX; 
	else
		return _ACT_NONE; 
}


string TypeDefs::RegulaName(const ERegula eRegula)
{
	switch(eRegula)
	{
		case _REGULA_L1:
			return "L1";
		case _REGULA_L2:
			return "L2";
		default:
			break;  
	}
	return "none";
}


ERegula TypeDefs::RegulaType(const char* sRegulaName)
{
	if(strcmp(sRegulaName, "L1") == 0)
		return _REGULA_L1; 
	else if(strcmp(sRegulaName, "L2") == 0)
		return _REGULA_L2; 
	else
		return _REGULA_NONE;
}


void TypeDefs::Print_PerceptronLearningParamsT(ostream& os, const PerceptronLearningParamsT perceptronLearningParamsT)
{
	os<<"Regula:"<<RegulaName(perceptronLearningParamsT.regula)<<endl; 
	os<<"MiniBatch:"<<perceptronLearningParamsT.mini_batch<<endl; 
	os<<"Iterations:"<<perceptronLearningParamsT.iterations<<endl; 
	os<<"LearningRate:"<<perceptronLearningParamsT.learning_rate<<endl; 
	os<<"RateDecay:"<<perceptronLearningParamsT.rate_decay<<endl; 
	os<<"Epsilon:"<<perceptronLearningParamsT.epsilon<<endl; 
	if(perceptronLearningParamsT.batch_norm)
		os<<"BatchNorm:true"<<endl; 
	else
		os<<"BatchNorm:false"<<endl; 
}


bool TypeDefs::Read_PerceptronLearningParamsT(PerceptronLearningParamsT& perceptronLearningParamsT, istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 6)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "Regula")
			perceptronLearningParamsT.regula = RegulaType(ar.GetString(1).c_str()); 
		else if(ar.GetString(0) == "MiniBatch")
			sscanf(ar.GetString(1).c_str(), "%d", &(perceptronLearningParamsT.mini_batch));
		else if(ar.GetString(0) == "Iterations")
			sscanf(ar.GetString(1).c_str(), "%d", &(perceptronLearningParamsT.iterations));
		else if(ar.GetString(0) == "LearningRate")
			sscanf(ar.GetString(1).c_str(), "%lf", &(perceptronLearningParamsT.learning_rate));
		else if(ar.GetString(0) == "RateDecay")
			sscanf(ar.GetString(1).c_str(), "%lf", &(perceptronLearningParamsT.rate_decay));
		else if(ar.GetString(0) == "Epsilon")
			sscanf(ar.GetString(1).c_str(), "%lf", &(perceptronLearningParamsT.epsilon));
		else if(ar.GetString(0) == "BatchNorm")
		{
			if(ar.GetString(1) == "true")
				perceptronLearningParamsT.batch_norm = true; 
			else
				perceptronLearningParamsT.batch_norm = false; 
		}
		else
			return false;
		cnt++; 
	}
	return true; 
}


string TypeDefs::ToString_PerceptronLearningParamsT(const PerceptronLearningParamsT perceptronLearningParamsT)
{
	char stmp[128];
	sprintf(stmp, "%s,%d,%d,%.6g,%.6g,%.6g", 
			RegulaName(perceptronLearningParamsT.regula).c_str(),
			perceptronLearningParamsT.mini_batch,
			perceptronLearningParamsT.iterations,
			perceptronLearningParamsT.learning_rate,
			perceptronLearningParamsT.rate_decay,
			perceptronLearningParamsT.epsilon);
	string str = stmp; 
	if(perceptronLearningParamsT.batch_norm)
		str += ",true";
	else
		str += ",false";
	return str; 
}


bool TypeDefs::FromString_PerceptronLearningParamsT(PerceptronLearningParamsT& perceptronLearningParamsT, const char* sStr)
{
	StringArray ar(sStr, ","); 
	if(ar.Count() != 7)
		return false; 
	perceptronLearningParamsT.regula = RegulaType(ar.GetString(0).c_str());
	sscanf(ar.GetString(1).c_str(), "%d", &(perceptronLearningParamsT.mini_batch));
	sscanf(ar.GetString(2).c_str(), "%d", &(perceptronLearningParamsT.iterations));
	sscanf(ar.GetString(3).c_str(), "%lf", &(perceptronLearningParamsT.learning_rate));
	sscanf(ar.GetString(4).c_str(), "%lf", &(perceptronLearningParamsT.rate_decay));
	sscanf(ar.GetString(5).c_str(), "%lf", &(perceptronLearningParamsT.epsilon));
	if(ar.GetString(6) == "true")
		perceptronLearningParamsT.batch_norm = true; 
	else
		perceptronLearningParamsT.batch_norm = false; 
	return true; 
}


void TypeDefs::Print_PerceptronParamsT(ostream& os, const PerceptronParamsT perceptronParamsT)
{
	os<<"Input:"<<perceptronParamsT.input-1<<endl; 
	os<<"Output:"<<perceptronParamsT.output<<endl; 
	os<<"Activation:"<<ActName(perceptronParamsT.act_output)<<endl; 
}


bool TypeDefs::Read_PerceptronParamsT(PerceptronParamsT& perceptronParamsT, istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 3)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2) 
			return false; 
		if(ar.GetString(0) == "Input")
		{
			sscanf(ar.GetString(1).c_str(), "%d", &(perceptronParamsT.input));
			perceptronParamsT.input += 1; // add 1 for bias nodes
		}
		else if(ar.GetString(0) == "Output")
			sscanf(ar.GetString(1).c_str(), "%d", &(perceptronParamsT.output));
		else if(ar.GetString(0) == "Activation")
			perceptronParamsT.act_output = ActType(ar.GetString(1).c_str());
		else
			return false;
		cnt++; 
	}
	return true; 
}


string TypeDefs::ToString_PerceptronParamsT(const PerceptronParamsT perceptronParamsT)
{
	char stmp[128];
	sprintf(stmp, "%d,%d,%s", 
			perceptronParamsT.input - 1, 
			perceptronParamsT.output, 
			ActName(perceptronParamsT.act_output).c_str()); 
	return string(stmp); 
}


bool TypeDefs::FromString_PerceptronParamsT(PerceptronParamsT& perceptronParamsT, const char* sStr)
{
	StringArray ar(sStr, ","); 
	if(ar.Count() != 3)
		return false; 
	sscanf(ar.GetString(0).c_str(), "%d", &(perceptronParamsT.input)); 
	perceptronParamsT.input += 1; 
	sscanf(ar.GetString(1).c_str(), "%d", &(perceptronParamsT.output)); 
	perceptronParamsT.act_output = ActType(ar.GetString(2).c_str()); 
	return true; 
}


bool TypeDefs::IsEqual_PerceptronParamsT(const PerceptronParamsT perceptronParamsT_1, const PerceptronParamsT perceptronParamsT_2)
{
	if(perceptronParamsT_1.input != perceptronParamsT_2.input || 
			perceptronParamsT_1.output != perceptronParamsT_2.output || 
			perceptronParamsT_1.act_output != perceptronParamsT_2.act_output)
	{
		return false; 
	}
	return true; 
}


void TypeDefs::Print_MLPParamsT(ostream& os, const MLPParamsT mlpParamsT)
{
	os<<"Input:"<<mlpParamsT.input-1<<endl; 
	os<<"Output:"<<mlpParamsT.output<<endl; 
	for(size_t i = 0; i < mlpParamsT.vtr_hidden.size(); i++) 
	{
		if(i == 0)
			os<<"Hiddens:"<<mlpParamsT.vtr_hidden[i]; 
		else
			os<<","<<mlpParamsT.vtr_hidden[i]; 
	}
	os<<endl;
	os<<"ActHidden:"<<ActName(mlpParamsT.act_hidden)<<endl; 
	os<<"ActOutput:"<<ActName(mlpParamsT.act_output)<<endl; 
}


bool TypeDefs::Read_MLPParamsT(MLPParamsT& mlpParamsT, istream& is)
{
	int32_t cnt = 0, hidden;
	string str;
	while(cnt < 5)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2) 
			return false; 
		if(ar.GetString(0) == "Input")
		{
			sscanf(ar.GetString(1).c_str(), "%d", &(mlpParamsT.input));
			mlpParamsT.input += 1; // add 1 for bias nodes
		}
		else if(ar.GetString(0) == "Output")
			sscanf(ar.GetString(1).c_str(), "%d", &(mlpParamsT.output));
		else if(ar.GetString(0) == "Hiddens")
		{
			mlpParamsT.vtr_hidden.clear(); 
			StringArray array(ar.GetString(1).c_str(), ","); 
			for(int32_t i = 0; i < array.Count(); i++) 
			{
				sscanf(array.GetString(i).c_str(), "%d", &hidden); 
				mlpParamsT.vtr_hidden.push_back(hidden); 
			}
		}
		else if(ar.GetString(0) == "ActHidden")
			mlpParamsT.act_hidden = ActType(ar.GetString(1).c_str());
		else if(ar.GetString(0) == "ActOutput")
			mlpParamsT.act_output = ActType(ar.GetString(1).c_str());
		else
			return false;
		cnt++; 
	}
	return true; 
}


string TypeDefs::ToString_MLPParamsT(const MLPParamsT mlpParamsT)
{
	char stmp[128];
	sprintf(stmp, "%d,%d,%s,%s", 
			mlpParamsT.input, 	
			mlpParamsT.output, 	
			ActName(mlpParamsT.act_hidden).c_str(), 
			ActName(mlpParamsT.act_output).c_str()); 
	string str(stmp); 	
	for(size_t i = 0; i < mlpParamsT.vtr_hidden.size(); i++) 
	{
		sprintf(stmp, ",%d", mlpParamsT.vtr_hidden[i]); 
		str += stmp; 
	}
	return str; 
}


bool TypeDefs::FromString_MLPParamsT(MLPParamsT& mlpParamsT, const char* sStr)
{
	StringArray ar(sStr, ","); 
	if(ar.Count() < 5)
		return false; 
	sscanf(ar.GetString(0).c_str(), "%d", &(mlpParamsT.input)); 
	sscanf(ar.GetString(1).c_str(), "%d", &(mlpParamsT.output)); 
	mlpParamsT.act_hidden = ActType(ar.GetString(2).c_str());
	mlpParamsT.act_output = ActType(ar.GetString(3).c_str());
	mlpParamsT.vtr_hidden.clear();
	int32_t hidden; 
	for(int32_t i = 4; i < ar.Count(); i++) 
	{
		sscanf(ar.GetString(i).c_str(), "%d", &hidden); 
		mlpParamsT.vtr_hidden.push_back(hidden); 
	}
	return true; 
}


bool TypeDefs::IsEqual_MLPParamsT(const MLPParamsT mlpParamsT_1, const MLPParamsT mlpParamsT_2)
{
	if(mlpParamsT_1.input != mlpParamsT_2.input || 
			mlpParamsT_1.output != mlpParamsT_2.output || 
			mlpParamsT_1.act_hidden != mlpParamsT_2.act_hidden || 
			mlpParamsT_1.act_output != mlpParamsT_2.act_output) 
	{
		return false; 
	}
	if(mlpParamsT_1.vtr_hidden.size() != mlpParamsT_2.vtr_hidden.size())
		return false; 
	for(size_t i = 0; i < mlpParamsT_1.vtr_hidden.size(); i++) 
	{
		if(mlpParamsT_1.vtr_hidden[i] != mlpParamsT_2.vtr_hidden[i])
			return false; 
	}
	return true; 
}

