// predict_example.cpp
//
// example of Model
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v1.0 2016-05-31
//

#include <iostream>
#include <fstream>
using namespace std; 
#include "metis_uti.h"
using namespace metis_uti; 
#include "metis_nn.h"
using namespace metis_nn; 
#include "metis_pred.h"
using namespace metis_pred; 
#include <stdio.h>
#include <string.h>
#include <math.h>


bool ReadPatterns(vector<Pattern*>& vtrPatts, const char* sPattFile)
{
	ifstream ifs(sPattFile);
	if(!ifs.is_open())
		return false; 
	string str; 
	Pattern* ppatt = NULL; 
	while(!ifs.eof())
	{
		std::getline(ifs, str); 
		if(str.empty())
			continue; 
		ppatt = new Pattern(); 	
		if(!ppatt->FromString(str.c_str()))
		{
			delete ppatt; 
			continue; 
		}
		vtrPatts.push_back(ppatt); 
	}
	ifs.close(); 
	if(vtrPatts.empty())
		return false; 
	return true; 
}



void DoPredict(double* y, const int32_t y_len, Pattern* pPatt, Model* pModel)
{
	vector<pair<int32_t,double> > vtr_feat;
	for(int32_t i = 0; i < pPatt->m_nXCnt; i++)
	{
		if(pPatt->m_x[i] != 0.0)
			vtr_feat.push_back(pair<int32_t,double>(i, pPatt->m_x[i])); 
	}

	for(int32_t t = 0; t < y_len; t++) 
	{
		y[t] = pModel->Predict(vtr_feat, t); 
	}
}


int main(int argc, char** argv)
{
	if(argc != 3)
	{
		cout<<"Usage: "<<argv[0]<<" <model_file> <testing_patterns_file>"<<endl;
		return -1; 
	}
	Model* p_model = Model::LoadModel(argv[1]); 
	if(!p_model) 
	{
		cout<<"failed to load model from "<<argv[1]<<endl; 
		return -2; 
	}

	vector<Pattern*> vtr_patts; 
	if(!ReadPatterns(vtr_patts, argv[2]))
	{
		cout<<"failed to load patterns from "<<argv[2]<<endl; 
		return -3; 
	}

	int32_t y_len = vtr_patts[0]->m_nYCnt;	
	double* y = new double[y_len]; 
	int32_t patts = (int32_t)vtr_patts.size(); 
	int32_t correct = 0, success = 0; 
	double error;	
	double rmse = 0;
	Timer timer; 
	RocAnalyzer roc; 

	for(int32_t i = 0; i < patts; i++) 
	{
		timer.Start(); 
		DoPredict(y, y_len, vtr_patts[i], p_model); 
		timer.Stop();
		success += 1;  
		error = Pattern::Error(y, vtr_patts[i]->m_y, y_len);
		rmse += error;
		if(vtr_patts[i]->m_y[0] == 1)
			roc.Insert(_POSITIVE, y[0]); 
		else
			roc.Insert(_NEGATIVE, y[0]); 

		if(y_len == 1)
		{
			if(fabs(y[0] - vtr_patts[i]->m_y[0]) < 0.5)	
				correct += 1;
			else
				printf("** ");
		}
		else
		{
			if(Pattern::MaxOff(y, y_len) == Pattern::MaxOff(vtr_patts[i]->m_y, vtr_patts[i]->m_nYCnt))
				correct += 1;
			else
				printf("** ");
		}
		printf("(%d) [%s] -> [%s] | error: %.12g | time_cost(ms): %.3f \n", 
				i+1, 
				Pattern::ArrayToString(vtr_patts[i]->m_y, vtr_patts[i]->m_nYCnt).c_str(), 
				Pattern::ArrayToString(y, y_len).c_str(), 
				error, 
				timer.GetLast_asMSec());
		delete vtr_patts[i]; 
	}
	rmse = sqrt(rmse / (double)success); 
	vtr_patts.clear(); 

	printf("Recall: %.4g%%, Precision: %.4g%%, RMSE: %.6g, AUC: %.6g\n", 
			(double)(success * 100) / (double)patts, 
			(double)(correct * 100) / (double)(success), 
			rmse, roc.Auc()); 

	delete p_model; 
	return 0; 
}



