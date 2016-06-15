#ifndef _METIS_UTILITY_ROC_ANALYZER_H
#define _METIS_UTILITY_ROC_ANALYZER_H

#include <vector>
#include <map>
#include <algorithm>
using namespace std; 
#include <stdint.h>


namespace metis_uti
{

enum ELabelType
{
	_POSITIVE,
	_NEGATIVE
};

enum EPNDetType
{
	_TP,
	_FN,
	_FP,
	_TN
}; 


// CLASS
// 	PNDetTable - Positive & Negative determination table
//
// DESCRIPTION
//  +----------+-------------------------------------------+
//  |          |                 Prediction                |
//  |          +---------------------+---------------------+
//  |          |     P(positive)     |     N(negative)     |
//  |------+---+---------------------+---------------------+
//  |      | P | TP (true positive)  | FN (false negative) |
//  | Real +---+---------------------+---------------------+
//  |      | N | FP (false positive) | TN (true negative)  |
//  +------+---+---------------------+---------------------+
class PNDetTable
{
public: 
	PNDetTable(); 
	PNDetTable(const PNDetTable& table); 
	virtual ~PNDetTable();

	void Clear(); 

	PNDetTable& operator = (const PNDetTable& table); 

	int32_t SampleCnt(); 

	// TP + FN
	int32_t RealPositiveCnt(); 
	// TP + FP
	int32_t PredPositiveCnt();

	// FP + TN
	int32_t RealNegativeCnt(); 
	// FN + TN
	int32_t PredNegativeCnt(); 

	// TPR = TP / (TP + FN)
	double TPR(); 
	// FPR = FP / (FP + TN)
	double FPR(); 
	
	// TNR = TN / (FP + TN) 
	double TNR();
	// FNR = FN / (TP + FN)
	double FNR();

	// PosPrecision = TP / (TP + FP)
	double PosPrecision(); 
	// NegPrecision = TN / (FN + TN)
	double NegPrecision(); 
	
	// PosRecall = TPR = TP / (TP + FN)
	double PosRecall(); 
	// NegRecall = TNR = TN / (FP + TN)
	double NegRecall(); 

	// PosF1 = 2 * PosPrecision * PosRecall / (PosPrecision + PosRecall)
	double PosF1(); 
	// NegF1 = 2 * NegPrecision * NegRecall / (NegPrecision + NegRecall)
	double NegF1(); 

public: 
	int32_t m_nCnt[4]; 
}; 



// CLASS
//	RocAnalyzer - ROC(Receiver Operating Characteristic) analyser for classification 
class RocAnalyzer
{
public:
	RocAnalyzer(const double dScoreFrom = 0.0, const double dScoreTo = 1.0); 
	virtual ~RocAnalyzer();

	void Clear(); 

	void Insert(const ELabelType realLabel,  const double dScore); 

	double Auc(); 

	void RocCurve(vector<pair<double, PNDetTable> >& curve, const double dStep = 0.01); 

	int32_t Count(const ELabelType eType); 

private:
	static bool Cmp(pair<ELabelType,double> a, pair<ELabelType, double> b); 

private: 
	double m_dScoreFrom;
	double m_dScoreTo;

	int32_t m_nPosCnt;
	int32_t m_nNegCnt;
	
	vector<pair<ELabelType,double> > m_vtrLabelScore;
}; 


}

#endif /* _METIS_UTILITY_ROC_ANALYZER_H */ 


