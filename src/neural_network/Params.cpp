#include "Params.h"
using namespace metis_nn; 
#include "StringArray.h"
#include "Config.h"
using namespace metis_uti; 
#include <string.h>
#include <stdio.h>



/////////////////////////////////////////////////////////////////////////////////
// class ActConv

ActConv::ActConv() 
{
}


ActConv::~ActConv()
{
}


string ActConv::ActName(const EActType eActType)
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
		case _ACT_RELU6:
			return "relu6";
		case _ACT_SOFTMAX: 
			return "softmax";
		default:
			break;
	}
	return "none";
}


EActType ActConv::ActType(const char* sActTypeName)
{
	if(strcmp(sActTypeName, "linear") == 0)
		return _ACT_LINEAR; 
	else if(strcmp(sActTypeName, "sigmoid") == 0)
		return _ACT_SIGMOID; 
	else if(strcmp(sActTypeName, "tanh") == 0)
		return _ACT_TANH; 
	else if(strcmp(sActTypeName, "relu") == 0)
		return _ACT_RELU; 
	else if(strcmp(sActTypeName, "relu6") == 0)
		return _ACT_RELU6; 
	else if(strcmp(sActTypeName, "softmax") == 0)
		return _ACT_SOFTMAX; 
	else
		return _ACT_NONE; 
}


/////////////////////////////////////////////////////////////////////////////////
// class RegulaConv

RegulaConv::RegulaConv()
{
}


RegulaConv::~RegulaConv()
{
}


string RegulaConv::RegulaName(const ERegula eRegula)
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


ERegula RegulaConv::RegulaType(const char* sRegulaName)
{
	if(strcmp(sRegulaName, "L1") == 0)
		return _REGULA_L1; 
	else if(strcmp(sRegulaName, "L2") == 0)
		return _REGULA_L2; 
	else
		return _REGULA_NONE;
}


/////////////////////////////////////////////////////////////////////////////////
// class OptimConv

OptimConv::OptimConv()
{
}


OptimConv::~OptimConv()
{
}


string OptimConv::OptimName(const EOptim eOptim)
{
	switch(eOptim)
	{
		case _OPTIM_SGD:
			return "SGD";
		case _OPTIM_MOMENTUM:
			return "Momentum";
		case _OPTIM_NAG:
			return "NAG";
		case _OPTIM_ADAGRAD:
			return "AdaGrad";
		case _OPTIM_RMSPROP:
			return "RMSprop";
		case _OPTIM_ADADELTA:
			return "AdaDelta";
		case _OPTIM_ADAM:
			return "Adam";
		default:
			break; 
	}
	return "NAG";
}


EOptim OptimConv::OptimType(const char* sOptimName)
{
	if(strcmp(sOptimName, "SGD") == 0)
		return _OPTIM_SGD; 
	else if(strcmp(sOptimName, "Momentum") == 0)
		return _OPTIM_MOMENTUM; 
	else if(strcmp(sOptimName, "NAG") == 0)
		return _OPTIM_NAG; 
	else if(strcmp(sOptimName, "AdaGrad") == 0)
		return _OPTIM_ADAGRAD; 
	else if(strcmp(sOptimName, "RMSprop") == 0)
		return _OPTIM_RMSPROP; 
	else if(strcmp(sOptimName, "AdaDelta") == 0)
		return _OPTIM_ADADELTA; 
	else if(strcmp(sOptimName, "Adam") == 0)
		return _OPTIM_ADAM; 
	else
		return _OPTIM_NAG; 
}


/////////////////////////////////////////////////////////////////////////////////
// class OptimParams

OptimParams::OptimParams() : learning_rate_init(0.001)
{
}


OptimParams::~OptimParams()
{
}


OptimParams* OptimParams::New(const EOptim eOptim)
{
	OptimParams* p = NULL; 
	switch(eOptim)
	{
		case _OPTIM_SGD:
			p = new SGDParams();
			break;  
		case _OPTIM_MOMENTUM: 
			p = new MomentumParams();
			break;  
		case _OPTIM_NAG: 
			p = new NAGParams();
			break;  
		case _OPTIM_ADAGRAD: 
			p = new AdaGradParams();
			break;  
		case _OPTIM_RMSPROP: 
			p = new RMSpropParams();
			break;  
		case _OPTIM_ADADELTA: 
			p = new AdaDeltaParams();
			break;  
		case _OPTIM_ADAM: 
			p = new AdamParams();
			break;  
		default: 
			break;  
	}
	return p; 
}



/////////////////////////////////////////////////////////////////////////////////
// class SGDParams

SGDParams::SGDParams() : OptimParams(), regula(_REGULA_L2), lambda(0.01)
{
}


SGDParams::SGDParams(const SGDParams& params) : OptimParams()
{
	regula = params.regula; 
	lambda = params.lambda; 
	learning_rate_init = params.learning_rate_init; 
}


SGDParams::~SGDParams()
{
}


SGDParams& SGDParams::operator = (const SGDParams& params)
{
	regula = params.regula; 
	lambda = params.lambda; 
	learning_rate_init = params.learning_rate_init; 
	return *this; 
}


void SGDParams::Print(ostream& os)
{
	os<<"SGD_Regula:"<<RegulaConv::RegulaName(regula)<<endl; 
	os<<"SGD_Lambda:"<<lambda<<endl; 
	os<<"SGD_LearningRateInit:"<<learning_rate_init<<endl; 
}


bool SGDParams::Read(istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 3)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "SGD_Regula")
			regula = RegulaConv::RegulaType(ar.GetString(1).c_str());
		else if(ar.GetString(0) == "SGD_Lambda")
			sscanf(ar.GetString(1).c_str(), "%lf", &lambda);  
		else if(ar.GetString(0) == "SGD_LearningRateInit")
			sscanf(ar.GetString(1).c_str(), "%lf", &learning_rate_init);  
		else
			return false;
		cnt++; 
	}
	return true;  
}


string SGDParams::ToString()
{
	char stmp[128]; 
	sprintf(stmp, "%s_%.12g_%.12g", 
			RegulaConv::RegulaName(regula).c_str(),
			lambda,
			learning_rate_init); 
	return string(stmp);  
}


bool SGDParams::FromString(const char* sStr)
{
	StringArray ar(sStr, "_"); 
	if(ar.Count() != 3)
		return false; 
	regula = RegulaConv::RegulaType(ar.GetString(0).c_str());
	sscanf(ar.GetString(1).c_str(), "%lf", &lambda);  
	sscanf(ar.GetString(2).c_str(), "%lf", &learning_rate_init);  
	return true; 
}


bool SGDParams::FromConfig(const char* sConfigFile)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	if(conf.ValCnt("SGD_Regula") > 0)
		regula = RegulaConv::RegulaType(conf.GetVal_asString("SGD_Regula").c_str());
	if(conf.ValCnt("SGD_Lambda") > 0)
		lambda = conf.GetVal_asFloat("SGD_Lambda"); 
	if(conf.ValCnt("SGD_LearningRateInit") > 0)
		learning_rate_init = conf.GetVal_asFloat("SGD_LearningRateInit"); 
	return true; 
}


/////////////////////////////////////////////////////////////////////////////////
// class MomentumParams

MomentumParams::MomentumParams() : OptimParams(), beta(0.9)
{
}


MomentumParams::MomentumParams(const MomentumParams& params) : OptimParams()
{
	beta = params.beta; 
	learning_rate_init = params.learning_rate_init; 
}


MomentumParams::~MomentumParams()
{
}


MomentumParams& MomentumParams::operator = (const MomentumParams& params)
{
	beta = params.beta; 
	learning_rate_init = params.learning_rate_init; 
	return *this; 
}


void MomentumParams::Print(ostream& os)
{
	os<<"Momentum_Beta:"<<beta<<endl; 
	os<<"Momentum_LearningRateInit:"<<learning_rate_init<<endl; 
}


bool MomentumParams::Read(istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 2)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "Momentum_Beta")
			sscanf(ar.GetString(1).c_str(), "%lf", &beta);  
		else if(ar.GetString(0) == "Momentum_LearningRateInit")
			sscanf(ar.GetString(1).c_str(), "%lf", &learning_rate_init);  
		else
			return false;
		cnt++; 
	}
	return true;  
}


string MomentumParams::ToString()
{
	char stmp[128]; 
	sprintf(stmp, "%.12g_%.12g", beta, learning_rate_init); 
	return string(stmp); 
}


bool MomentumParams::FromString(const char* sStr)
{
	StringArray ar(sStr, "_"); 
	if(ar.Count() != 2)
		return false; 
	sscanf(ar.GetString(0).c_str(), "%lf", &beta);  
	sscanf(ar.GetString(1).c_str(), "%lf", &learning_rate_init);  
	return true; 
}


bool MomentumParams::FromConfig(const char* sConfigFile)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	if(conf.ValCnt("Momentum_Beta") > 0)
		beta = conf.GetVal_asFloat("Momentum_Beta"); 
	if(conf.ValCnt("Momentum_LearningRateInit") > 0)
		learning_rate_init = conf.GetVal_asFloat("Momentum_LearningRateInit"); 
	return true; 
}


/////////////////////////////////////////////////////////////////////////////////
// class NAGParams

NAGParams::NAGParams() : OptimParams(), beta(0.9)
{
}


NAGParams::NAGParams(const NAGParams& params) : OptimParams()
{
	beta = params.beta;
	learning_rate_init = params.learning_rate_init; 
}


NAGParams::~NAGParams()
{
}


NAGParams& NAGParams::operator = (const NAGParams& params) 
{
	beta = params.beta;
	learning_rate_init = params.learning_rate_init; 
	return *this; 
}


void NAGParams::Print(ostream& os)
{
	os<<"NAG_Beta:"<<beta<<endl; 
	os<<"NAG_LearningRateInit:"<<learning_rate_init<<endl; 
}


bool NAGParams::Read(istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 2)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "NAG_Beta")
			sscanf(ar.GetString(1).c_str(), "%lf", &beta);  
		else if(ar.GetString(0) == "NAG_LearningRateInit")
			sscanf(ar.GetString(1).c_str(), "%lf", &learning_rate_init);  
		else
			return false;
		cnt++; 
	}
	return true;  
}


string NAGParams::ToString()
{
	char stmp[128]; 
	sprintf(stmp, "%.12g_%.12g", beta, learning_rate_init); 
	return string(stmp); 
}


bool NAGParams::FromString(const char* sStr)
{
	StringArray ar(sStr, "_"); 
	if(ar.Count() != 2)
		return false; 
	sscanf(ar.GetString(0).c_str(), "%lf", &beta);  
	sscanf(ar.GetString(1).c_str(), "%lf", &learning_rate_init);  
	return true; 
}


bool NAGParams::FromConfig(const char* sConfigFile)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	if(conf.ValCnt("NAG_Beta") > 0)
		beta = conf.GetVal_asFloat("NAG_Beta"); 
	if(conf.ValCnt("NAG_LearningRateInit") > 0)
		learning_rate_init = conf.GetVal_asFloat("NAG_LearningRateInit"); 
	return true; 
}


/////////////////////////////////////////////////////////////////////////////////
// class AdaGradParams

AdaGradParams::AdaGradParams() : OptimParams(), eps(1e-6)
{
}


AdaGradParams::AdaGradParams(const AdaGradParams& params) : OptimParams()
{
	eps = params.eps;
	learning_rate_init = params.learning_rate_init; 
}


AdaGradParams::~AdaGradParams()
{
}


AdaGradParams& AdaGradParams::operator = (const AdaGradParams& params)
{
	eps = params.eps;
	learning_rate_init = params.learning_rate_init; 
	return *this; 
}


void AdaGradParams::Print(ostream& os)
{
	os<<"AdaGrad_Eps:"<<eps<<endl; 
	os<<"AdaGrad_LearningRateInit:"<<learning_rate_init<<endl; 
}


bool AdaGradParams::Read(istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 2)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "AdaGrad_Eps")
			sscanf(ar.GetString(1).c_str(), "%lf", &eps);  
		else if(ar.GetString(0) == "AdaGrad_LearningRateInit")
			sscanf(ar.GetString(1).c_str(), "%lf", &learning_rate_init);  
		else
			return false;
		cnt++; 
	}
	return true;  
}


string AdaGradParams::ToString()
{
	char stmp[128]; 
	sprintf(stmp, "%.12g_%.12g", 
			eps,
			learning_rate_init);
	return string(stmp); 
}


bool AdaGradParams::FromString(const char* sStr)
{
	StringArray ar(sStr, "_"); 
	if(ar.Count() != 2)
		return false; 
	sscanf(ar.GetString(0).c_str(), "%lf", &eps);  
	sscanf(ar.GetString(1).c_str(), "%lf", &learning_rate_init);  
	return true; 
}


bool AdaGradParams::FromConfig(const char* sConfigFile)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	if(conf.ValCnt("AdaGrad_Eps") > 0)
		eps = conf.GetVal_asFloat("AdaGrad_Eps"); 
	if(conf.ValCnt("AdaGrad_LearningRateInit") > 0)
		learning_rate_init = conf.GetVal_asFloat("AdaGrad_LearningRateInit"); 
	return true; 
}


/////////////////////////////////////////////////////////////////////////////////
// class RMSpropParams

RMSpropParams::RMSpropParams() : OptimParams(), beta(0.9), eps(1e-6)
{
}


RMSpropParams::RMSpropParams(const RMSpropParams& params) : OptimParams()
{
	beta = params.beta; 
	eps = params.eps; 
	learning_rate_init = params.learning_rate_init; 
}


RMSpropParams::~RMSpropParams()
{
}


RMSpropParams& RMSpropParams::operator = (const RMSpropParams& params)
{
	beta = params.beta; 
	eps = params.eps; 
	learning_rate_init = params.learning_rate_init; 
	return *this; 
}


void RMSpropParams::Print(ostream& os)
{
	os<<"RMSprop_Beta:"<<beta<<endl; 
	os<<"RMSprop_Eps:"<<eps<<endl; 
	os<<"RMSprop_LearningRateInit:"<<learning_rate_init<<endl; 
}


bool RMSpropParams::Read(istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 3)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "RMSprop_Beta")
			sscanf(ar.GetString(1).c_str(), "%lf", &beta);  
		else if(ar.GetString(0) == "RMSprop_Eps")
			sscanf(ar.GetString(1).c_str(), "%lf", &eps);  
		else if(ar.GetString(0) == "RMSprop_LearningRateInit")
			sscanf(ar.GetString(1).c_str(), "%lf", &learning_rate_init);  
		else
			return false;
		cnt++; 
	}
	return true;  
}


string RMSpropParams::ToString()
{
	char stmp[128]; 
	sprintf(stmp, "%.12g_%.12g_%.12g", 
			beta, 
			eps,
			learning_rate_init);
	return string(stmp); 
}


bool RMSpropParams::FromString(const char* sStr)
{
	StringArray ar(sStr, "_"); 
	if(ar.Count() != 3)
		return false; 
	sscanf(ar.GetString(0).c_str(), "%lf", &beta);  
	sscanf(ar.GetString(1).c_str(), "%lf", &eps);  
	sscanf(ar.GetString(2).c_str(), "%lf", &learning_rate_init);  
	return true; 
}


bool RMSpropParams::FromConfig(const char* sConfigFile)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	if(conf.ValCnt("RMSprop_Beta") > 0)
		beta = conf.GetVal_asFloat("RMSprop_Beta"); 
	if(conf.ValCnt("RMSprop_Eps") > 0)
		eps = conf.GetVal_asFloat("RMSprop_Eps"); 
	if(conf.ValCnt("RMSprop_LearningRateInit") > 0)
		learning_rate_init = conf.GetVal_asFloat("RMSprop_LearningRateInit"); 
	return true; 
}



/////////////////////////////////////////////////////////////////////////////////
// class AdaDeltaParams

AdaDeltaParams::AdaDeltaParams() : OptimParams(), rho(0.9), beta(0.9), eps(1e-8)
{
}


AdaDeltaParams::AdaDeltaParams(const AdaDeltaParams& params) : OptimParams()
{
	rho = params.rho;
	beta = params.beta;
	eps = params.eps; 
	learning_rate_init = params.learning_rate_init; 
}


AdaDeltaParams::~AdaDeltaParams()
{
}


AdaDeltaParams& AdaDeltaParams::operator = (const AdaDeltaParams& params)
{
	rho = params.rho;
	beta = params.beta;
	eps = params.eps; 
	learning_rate_init = params.learning_rate_init; 
	return *this; 
}


void AdaDeltaParams::Print(ostream& os)
{
	os<<"AdaDelta_Rho:"<<rho<<endl; 
	os<<"AdaDelta_Beta:"<<beta<<endl; 
	os<<"AdaDelta_Eps:"<<eps<<endl; 
}


bool AdaDeltaParams::Read(istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 3)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "AdaDelta_Rho")
			sscanf(ar.GetString(1).c_str(), "%lf", &rho);  
		else if(ar.GetString(0) == "AdaDelta_Beta")
			sscanf(ar.GetString(1).c_str(), "%lf", &beta);  
		else if(ar.GetString(0) == "AdaDelta_Eps")
			sscanf(ar.GetString(1).c_str(), "%lf", &eps);  
		else
			return false;
		cnt++; 
	}
	return true;  
}


string AdaDeltaParams::ToString()
{
	char stmp[128]; 
	sprintf(stmp, "%.12g_%.12g_%.12g", 
			rho,
			beta, 
			eps); 
	return string(stmp); 
}


bool AdaDeltaParams::FromString(const char* sStr)
{
	StringArray ar(sStr, "_"); 
	if(ar.Count() != 3)
		return false; 
	sscanf(ar.GetString(0).c_str(), "%lf", &rho);  
	sscanf(ar.GetString(1).c_str(), "%lf", &beta);  
	sscanf(ar.GetString(2).c_str(), "%lf", &eps);  
	return true; 
}


bool AdaDeltaParams::FromConfig(const char* sConfigFile)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	if(conf.ValCnt("AdaDelta_Rho") > 0)
		rho = conf.GetVal_asFloat("AdaDelta_Rho"); 
	if(conf.ValCnt("AdaDelta_Beta") > 0)
		beta = conf.GetVal_asFloat("AdaDelta_Beta"); 
	if(conf.ValCnt("AdaDelta_Eps") > 0)
		eps = conf.GetVal_asFloat("AdaDelta_Eps"); 
	return true; 
}


/////////////////////////////////////////////////////////////////////////////////
// class AdamParams

AdamParams::AdamParams() : OptimParams(), beta1(0.9), beta2(0.999), eps(1e-8)
{
}


AdamParams::AdamParams(const AdamParams& params) : OptimParams()
{
	beta1 = params.beta1; 
	beta2 = params.beta2; 
	eps = params.eps; 
	learning_rate_init = params.learning_rate_init; 
}


AdamParams::~AdamParams()
{
}


AdamParams& AdamParams::operator = (const AdamParams& params)
{
	beta1 = params.beta1; 
	beta2 = params.beta2; 
	eps = params.eps; 
	learning_rate_init = params.learning_rate_init; 
	return *this; 
}


void AdamParams::Print(ostream& os)
{
	os<<"Adam_Beta1:"<<beta1<<endl; 
	os<<"Adam_Beta2:"<<beta2<<endl; 
	os<<"Adam_Eps:"<<eps<<endl; 
	os<<"Adam_LearningRateInit:"<<learning_rate_init<<endl; 
}


bool AdamParams::Read(istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 4)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "Adam_Beta1")
			sscanf(ar.GetString(1).c_str(), "%lf", &beta1);  
		else if(ar.GetString(0) == "Adam_Beta2")
			sscanf(ar.GetString(1).c_str(), "%lf", &beta2);  
		else if(ar.GetString(0) == "Adam_Eps")
			sscanf(ar.GetString(1).c_str(), "%lf", &eps);  
		else if(ar.GetString(0) == "Adam_LearningRateInit")
			sscanf(ar.GetString(1).c_str(), "%lf", &learning_rate_init);  
		else
			return false;
		cnt++; 
	}
	return true;  
}


string AdamParams::ToString()
{
	char stmp[128]; 
	sprintf(stmp, "%.12g_%.12g_%.12g_%.12g", 
			beta1, 
			beta2, 
			eps,
			learning_rate_init);
	return string(stmp); 
}


bool AdamParams::FromString(const char* sStr)
{
	StringArray ar(sStr, "_"); 
	if(ar.Count() != 4)
		return false; 
	sscanf(ar.GetString(0).c_str(), "%lf", &beta1);  
	sscanf(ar.GetString(1).c_str(), "%lf", &beta2);  
	sscanf(ar.GetString(2).c_str(), "%lf", &eps);  
	sscanf(ar.GetString(3).c_str(), "%lf", &learning_rate_init);  
	return true; 
}


bool AdamParams::FromConfig(const char* sConfigFile)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	if(conf.ValCnt("Adam_Beta1") > 0)
		beta1 = conf.GetVal_asFloat("Adam_Beta1"); 
	if(conf.ValCnt("Adam_Beta2") > 0)
		beta2 = conf.GetVal_asFloat("Adam_Beta2"); 
	if(conf.ValCnt("Adam_Eps") > 0)
		eps = conf.GetVal_asFloat("Adam_Eps"); 
	if(conf.ValCnt("Adam_LearningRateInit") > 0)
		learning_rate_init = conf.GetVal_asFloat("Adam_LearningRateInit"); 
	return true; 
}


/////////////////////////////////////////////////////////////////////////////////
// class LearnParams

LearnParams::LearnParams() : optim(_OPTIM_NAG), batch_size(200), max_epoches(200), early_stop(10), epsilon(0.1), p_optim_params(NULL)
{
	p_optim_params = OptimParams::New(optim); 
}


LearnParams::LearnParams(const LearnParams& learnParams) 
{
	optim = learnParams.optim; 
	batch_size = learnParams.batch_size; 
	max_epoches = learnParams.max_epoches; 
	early_stop = learnParams.early_stop; 
	epsilon = learnParams.epsilon; 
	//////
	if(epsilon == _OPTIM_SGD)
		p_optim_params = new SGDParams(*((SGDParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_MOMENTUM)
		p_optim_params = new MomentumParams(*((MomentumParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_NAG)
		p_optim_params = new NAGParams(*((NAGParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_ADAGRAD)
		p_optim_params = new AdaGradParams(*((AdaGradParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_RMSPROP)
		p_optim_params = new RMSpropParams(*((RMSpropParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_ADADELTA)
		p_optim_params = new AdaDeltaParams(*((AdaDeltaParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_ADAM)
		p_optim_params = new AdamParams(*((AdamParams*)learnParams.p_optim_params)); 
}



LearnParams::~LearnParams()
{
	if(p_optim_params)
	{
		delete p_optim_params; 
		p_optim_params = NULL; 
	}
}

LearnParams& LearnParams::operator = (const LearnParams& learnParams)
{
	optim = learnParams.optim; 
	batch_size = learnParams.batch_size; 
	max_epoches = learnParams.max_epoches; 
	early_stop = learnParams.early_stop; 
	epsilon = learnParams.epsilon; 
	//////
	if(p_optim_params)
	{
		delete p_optim_params; 
		p_optim_params = NULL; 
	}
	if(epsilon == _OPTIM_SGD)
		p_optim_params = new SGDParams(*((SGDParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_MOMENTUM)
		p_optim_params = new MomentumParams(*((MomentumParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_NAG)
		p_optim_params = new NAGParams(*((NAGParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_ADAGRAD)
		p_optim_params = new AdaGradParams(*((AdaGradParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_RMSPROP)
		p_optim_params = new RMSpropParams(*((RMSpropParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_ADADELTA)
		p_optim_params = new AdaDeltaParams(*((AdaDeltaParams*)learnParams.p_optim_params)); 
	else if(epsilon == _OPTIM_ADAM)
		p_optim_params = new AdamParams(*((AdamParams*)learnParams.p_optim_params)); 
	return *this; 
}


void LearnParams::Print(ostream& os)
{
	os<<"Optim:"<<OptimConv::OptimName(optim)<<endl; 
	p_optim_params->Print(os); 
	os<<"BatchSize:"<<batch_size<<endl; 
	os<<"MaxEpoches:"<<max_epoches<<endl; 
	os<<"EarlyStop:"<<early_stop<<endl; 
	os<<"Epsilon:"<<epsilon<<endl; 
}


bool LearnParams::Read(istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 5)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "Optim")
		{
			optim = OptimConv::OptimType(ar.GetString(1).c_str());
			if(p_optim_params)
				delete p_optim_params; 
			p_optim_params = OptimParams::New(optim); 
			if(!p_optim_params->Read(is))
				return false;
		} 		
		else if(ar.GetString(0) == "BatchSize")
			sscanf(ar.GetString(1).c_str(), "%d", &batch_size);
		else if(ar.GetString(0) == "MaxEpoches")
			sscanf(ar.GetString(1).c_str(), "%d", &max_epoches); 
		else if(ar.GetString(0) == "EarlyStop")
			sscanf(ar.GetString(1).c_str(), "%d", &early_stop); 
		else if(ar.GetString(0) == "Epsilon")
			sscanf(ar.GetString(1).c_str(), "%lf", &epsilon); 
		else
			return false;
		cnt++; 
	}
	return true; 
}


string LearnParams::ToString()
{
	char stmp[256];
	sprintf(stmp, "%s,%s,%d,%d,%d,%.12g", 
			OptimConv::OptimName(optim).c_str(),
			p_optim_params->ToString().c_str(), 	
			batch_size, 
			max_epoches, 
			early_stop, 
			epsilon); 
	return string(stmp); 
}


bool LearnParams::FromString(const char* sStr)
{
	StringArray ar(sStr, ","); 
	if(ar.Count() != 6)
		return false; 
	optim = OptimConv::OptimType(ar.GetString(0).c_str());
	if(p_optim_params)
		delete p_optim_params;
	p_optim_params = OptimParams::New(optim);	
	if(!p_optim_params->FromString(ar.GetString(1).c_str()))
		return false; 
	sscanf(ar.GetString(2).c_str(), "%d", &batch_size);
	sscanf(ar.GetString(3).c_str(), "%d", &max_epoches); 
	sscanf(ar.GetString(4).c_str(), "%d", &early_stop); 
	sscanf(ar.GetString(5).c_str(), "%lf", &epsilon); 
	return true; 
}


bool LearnParams::FromConfig(const char* sConfigFile)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 

	if(conf.ValCnt("Optim") > 0)
	{
		optim = OptimConv::OptimType(conf.GetVal_asString("Optim").c_str()); 
		if(p_optim_params)
			delete p_optim_params;
		p_optim_params = OptimParams::New(optim);	
		if(!p_optim_params->FromConfig(sConfigFile))
			return false; 
	}		
	if(conf.ValCnt("BatchSize") > 0)
		batch_size = conf.GetVal_asInt("BatchSize"); 
	if(conf.ValCnt("MaxEpoches") > 0)
		max_epoches = conf.GetVal_asInt("MaxEpoches"); 
	if(conf.ValCnt("EarlyStop") > 0)
		early_stop = conf.GetVal_asInt("EarlyStop"); 
	if(conf.ValCnt("Epsilon") > 0)
		epsilon = conf.GetVal_asFloat("Epsilon"); 

	return true; 
}


/////////////////////////////////////////////////////////////////////////////////
// class ArchParams

ArchParams::ArchParams() : type(_ARCH_PARAMS_BASIC), input(0), output(0)
{
}


ArchParams::ArchParams(const ArchParams& archParams) 
{
	input = archParams.input; 
	output = archParams.output;  
	type = _ARCH_PARAMS_BASIC; 
}


ArchParams::~ArchParams()
{
}


ArchParams& ArchParams::operator = (const ArchParams& archParams)
{
	input = archParams.input; 
	output = archParams.output;  
	return *this; 
}


void ArchParams::Print(ostream& os)
{
	os<<"Input:"<<input<<endl; 
	os<<"Output:"<<output<<endl; 
}


bool ArchParams::Read(istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 2)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "Input")
			sscanf(ar.GetString(1).c_str(), "%d", &input); 
		else if(ar.GetString(0) == "Output")
			sscanf(ar.GetString(1).c_str(), "%d", &output); 
		else
			return false;
		cnt++; 
	}
	return true; 
}


string ArchParams::ToString()
{
	char stmp[128];
	sprintf(stmp, "%d,%d", input, output); 
	return string(stmp); 
}


bool ArchParams::FromString(const char* sStr)
{
	StringArray ar(sStr, ","); 
	if(ar.Count() != 2)
		return false; 
	sscanf(ar.GetString(0).c_str(), "%d", &input); 
	sscanf(ar.GetString(1).c_str(), "%d", &output); 
	return true; 
}


bool ArchParams::FromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 
	input = nInput; 
	output = nOutput; 
	return true; 
}


EArchParamsType ArchParams::GetType()
{
	return type; 
}


/////////////////////////////////////////////////////////////////////////////////
// class ArchParams_MLP

ArchParams_MLP::ArchParams_MLP() : ArchParams()
{
	type = _ARCH_PARAMS_MLP; 
}


ArchParams_MLP::ArchParams_MLP(const ArchParams_MLP& archParamsMLP) : ArchParams()
{
	input = archParamsMLP.input;
	output = archParamsMLP.output;
	hidden_act = archParamsMLP.hidden_act; 
	vtr_hiddens = archParamsMLP.vtr_hiddens;
}


ArchParams_MLP::~ArchParams_MLP()
{
}


ArchParams_MLP& ArchParams_MLP::operator = (const ArchParams_MLP& archParamsMLP)
{
	input = archParamsMLP.input;
	output = archParamsMLP.output;
	hidden_act = archParamsMLP.hidden_act; 
	vtr_hiddens = archParamsMLP.vtr_hiddens;
	return *this; 
}


void ArchParams_MLP::Print(ostream& os)
{
	os<<"Input:"<<input<<endl; 
	for(size_t h = 0; h < vtr_hiddens.size(); h++) 
	{
		if(h == 0)
			os<<"Hiddens:"<<vtr_hiddens[h];
		else
			os<<","<<vtr_hiddens[h];
	}
	os<<endl; 
	os<<"Output:"<<output<<endl; 
	os<<"ActHidden:"<<ActConv::ActName(hidden_act)<<endl; 
}


bool ArchParams_MLP::Read(istream& is)
{
	int32_t hidden, cnt = 0;
	string str;
	vtr_hiddens.clear();  
	while(cnt < 4)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "Input")
			sscanf(ar.GetString(1).c_str(), "%d", &input); 
		else if(ar.GetString(0) == "Hiddens")
		{
			StringArray ar_h(ar.GetString(1).c_str(), ","); 
			for(int32_t h = 0; h < ar_h.Count(); h++) 
			{
				sscanf(ar_h.GetString(h).c_str(), "%d", &hidden); 
				vtr_hiddens.push_back(hidden); 
			}
		}
		else if(ar.GetString(0) == "Output")
			sscanf(ar.GetString(1).c_str(), "%d", &output); 
		else if(ar.GetString(0) == "ActHidden")
			hidden_act = ActConv::ActType(ar.GetString(1).c_str()); 
		else
			return false;
		cnt++; 
	}
	return true; 
}


string ArchParams_MLP::ToString()
{
	char stmp[128]; 
	sprintf(stmp, "%d,%d,%s", input, output, ActConv::ActName(hidden_act).c_str()); 
	string str(stmp);
	for(size_t h = 0; h < vtr_hiddens.size(); h++) 
	{
		sprintf(stmp, ",%d", vtr_hiddens[h]); 
		str += stmp; 
	}
	return str;  
}


bool ArchParams_MLP::FromString(const char* sStr)
{
	vtr_hiddens.clear(); 
	StringArray ar(sStr, ","); 
	if(ar.Count() < 4)
		return false; 
	sscanf(ar.GetString(0).c_str(), "%d", &input); 
	sscanf(ar.GetString(1).c_str(), "%d", &output);
	hidden_act = ActConv::ActType(ar.GetString(2).c_str()); 
	int32_t hidden; 
	for(int32_t i = 3; i < ar.Count(); i++)
	{
		sscanf(ar.GetString(i).c_str(), "%d", &hidden);
		vtr_hiddens.push_back(hidden); 
	}
	return true; 
}


bool ArchParams_MLP::FromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput)
{
	Config conf; 
	if(!conf.Read(sConfigFile)) 
		return false; 

	if(conf.ValCnt("Hiddens") <= 0)
		return false; 	
	vtr_hiddens.clear(); 
	for(int32_t h = 0; h < conf.ValCnt("Hiddens"); h++) 
		vtr_hiddens.push_back(conf.GetVal_asInt("Hiddens", h)); 

	if(conf.ValCnt("ActHidden") <= 0)
		return false; 	
	hidden_act = ActConv::ActType(conf.GetVal_asString("ActHidden").c_str());

	input = nInput; 
	output = nOutput; 

	return true; 
}



