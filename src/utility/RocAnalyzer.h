// RocAnalyzer.h
//
// Analyzer in ROC method
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v1.0 2016-03-14
//

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
	// Construction & Destruction 
	PNDetTable(); 
	PNDetTable(const PNDetTable& table); 
	virtual ~PNDetTable();

	// Clear current table
	void Clear(); 

	// Reload assignment
	PNDetTable& operator = (const PNDetTable& table); 

	// Get count of samples
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
// 
// DESCRIPTION
//	RocAnalyzer supports ROC curve and AUC calculation 
//
class RocAnalyzer
{
public:
	// Construction & Destruction
	RocAnalyzer(const double dScoreFrom = 0.0, const double dScoreTo = 1.0); 
	virtual ~RocAnalyzer();

	// NAME
	//	Clear - clear current analyzer 	
	void Clear(); 

	// NAME
	//	Insert - intsert one predict result to current analyzer
	//	
	// DESCRIPTION
	//	realLabel: real category label of the sample
	//	dScore: predict score
	void Insert(const ELabelType realLabel,  const double dScore); 

	// NAME
	//	Auc - calculate the AUC (Area Under the Curve) value
	//
	// RETURN
	//	The AUC value
	double Auc(); 

	// NAME
	//	RocCurve - get the ROC curve paraments
	//
	// DESCRIPTION
	//	curve: out param, points of the curve
	//	dStep: step of threshold 
	void RocCurve(vector<pair<double, PNDetTable> >& curve, const double dStep = 0.01); 

	// NAME
	//	Count - get the sample count of positive or negative samples
	//
	// DESCRIPTION
	//	eType - _POSITIVE or _NEGATIVE
	//
	// RETURN
	//	Samples count
	int32_t Count(const ELabelType eType); 

private:
	// Compare two pair variables according to their score values (scondary value) 
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


