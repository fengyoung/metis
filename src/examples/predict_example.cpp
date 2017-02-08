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
#include "metis_pred.h"
using namespace metis_pred; 
#include <stdio.h>
#include <string.h>
#include <math.h>


void DoPredict(const char* sPattsFile, Model* pModel, const int32_t nTarget, const char* sRocFile = NULL)
{
	ifstream ifs(sPattsFile);
	if(!ifs.is_open())
	{
		cout<<"Failed to open patterns file "<<sPattsFile<<endl; 
		return; 
	}

	Pattern patt; 
	string str; 
	double pred_score[pModel->N_Output()], rmse; 
	int32_t tp = 0, fp = 0, fn = 0, tn = 0; 
	int32_t p_off, r_off; 
	Timer timer;  
	RocAnalyzer roc; 

	while(!ifs.eof())
	{
		std::getline(ifs, str); 
		if(str.empty())
			continue;
		// read pattern for testing
		if(!patt.FromString(str.c_str())) 
			continue;

		// IMPORTANT!!!
		// prediction, calculate pred_score of each label	
		timer.Start(); 
		for(int32_t i = 0; i < pModel->N_Output(); i++) 
			pred_score[i] = pModel->Predict(patt.m_x, patt.m_nXCnt, i); 
		timer.Stop(); 

		// effect evaluation and statistic	
		rmse = sqrt(Pattern::Error(pred_score, patt.m_y, patt.m_nYCnt)); 
		r_off = Pattern::MaxOff(patt.m_y, patt.m_nYCnt); 
		p_off = Pattern::MaxOff(pred_score, pModel->N_Output()); 
		if(r_off == nTarget && p_off == nTarget)
		{
			roc.Insert(_POSITIVE, pred_score[nTarget]); 
			tp += 1; 
		}
		else if(r_off != nTarget && p_off == nTarget)
		{
			printf("*** "); 
			roc.Insert(_NEGATIVE, pred_score[nTarget]); 
			fp += 1; 
		}
		else if(r_off == nTarget && p_off != nTarget)
		{
			printf("*** "); 
			roc.Insert(_POSITIVE, pred_score[nTarget]); 
			fn += 1; 
		}
		else
		{
			roc.Insert(_NEGATIVE, pred_score[nTarget]); 
			tn += 1; 
		}
		printf("(%d) [%s] -> [%s] | RMSE: %.6g | TimeCost(ms): %.3f\n", 
				tp + fp + fn + tn, 
				Pattern::ArrayToString(patt.m_y, patt.m_nYCnt).c_str(), 
				Pattern::ArrayToString(pred_score, pModel->N_Output()).c_str(),
				rmse, 
				timer.GetLast_asMSec()); 
	}

	ifs.close(); 

	double precision = (double)tp / (double)(tp + fp);  
	double recall = (double)tp / (double)(tp + fn);  
	printf("Precision: %.6g, Recall: %.6g, F1: %.6g, AUC: %.6g, AvgTimeCost(ms): %.3f\n", 
			precision, recall, (precision * recall * 2.0) / (precision + recall), roc.Auc(), timer.GetTotal_asMSec() / (double)(tp + fp + fn + tn)); 

	// save ROC curve
	if(sRocFile) 
	{
		FILE* fp = fopen(sRocFile, "w"); 
		if(fp)
		{
			vector<pair<double, PNDetTable> > curve; 
			roc.RocCurve(curve);

			fprintf(fp, "AUC;%.6g\n\n", roc.Auc()); 
			fprintf(fp, "Threshold;FPR;TPR;Pr;Recall;F1\n"); 
			for(size_t k = 0; k < curve.size(); k++) 
			{
				fprintf(fp, "%.6g;%.6g;%.6g;%.6g;%.6g;%.6g\n",
						curve[k].first,
						curve[k].second.FPR(),
						curve[k].second.TPR(),
						curve[k].second.PosPrecision(),
						curve[k].second.PosRecall(),
						curve[k].second.PosF1()); 
			}

			fclose(fp);
		}	
	}

}



int main(int argc, char** argv)
{
	if(argc < 4)
	{
		cout<<"Usage: "<<argv[0]<<" <model_file> <target_off> <testing_patterns_file> <|roc_file>"<<endl;
		return -1; 	
	}

	Model* p_model = Model::LoadModel(argv[1]); 
	if(!p_model) 
	{
		cout<<"Failed to load model from "<<argv[1]<<endl; 
		return -2; 
	}
	int32_t target_off; 
	sscanf(argv[2], "%d", &target_off);
	if(target_off < 0 || target_off >= p_model->N_Output())
	{
		cout<<"Target offset is out of boundary"<<endl; 
		return -3; 
	}

	if(argc == 4)
		DoPredict(argv[3], p_model, target_off); 
	else
		DoPredict(argv[3], p_model, target_off, argv[4]); 

	return 0; 
}



