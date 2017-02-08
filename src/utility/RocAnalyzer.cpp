#include "RocAnalyzer.h"
using namespace metis_uti; 
#include <iostream>
using namespace std; 


/////////////////////////////////////////////////////////////////////////////////////////////////////
// Class PNDetTable
/////////////////////////////////////////////////////////////////////////////////////////////////////


PNDetTable::PNDetTable()
{
	m_nCnt[_TP] = 0; 
	m_nCnt[_FN] = 0; 
	m_nCnt[_FP] = 0; 
	m_nCnt[_TN] = 0; 
}


PNDetTable::PNDetTable(const PNDetTable& table)
{
	m_nCnt[_TP] = table.m_nCnt[_TP]; 
	m_nCnt[_FN] = table.m_nCnt[_FN]; 
	m_nCnt[_FP] = table.m_nCnt[_FP]; 
	m_nCnt[_TN] = table.m_nCnt[_TN]; 
}


PNDetTable::~PNDetTable()
{
}


void PNDetTable::Clear()
{
	m_nCnt[_TP] = 0; 
	m_nCnt[_FN] = 0; 
	m_nCnt[_FP] = 0; 
	m_nCnt[_TN] = 0; 
}


PNDetTable& PNDetTable::operator = (const PNDetTable& table)
{
	m_nCnt[_TP] = table.m_nCnt[_TP]; 
	m_nCnt[_FN] = table.m_nCnt[_FN]; 
	m_nCnt[_FP] = table.m_nCnt[_FP]; 
	m_nCnt[_TN] = table.m_nCnt[_TN]; 
	return *this; 	
}


int32_t PNDetTable::SampleCnt()
{
	return (m_nCnt[_TP] + m_nCnt[_FP] + m_nCnt[_TN] + m_nCnt[_FN]); 
}


int32_t PNDetTable::RealPositiveCnt()
{
	return (m_nCnt[_TP] + m_nCnt[_FN]); 
}

int32_t PNDetTable::PredPositiveCnt()
{
	return (m_nCnt[_TP] + m_nCnt[_FP]); 
}


int32_t PNDetTable::RealNegativeCnt()
{
	return (m_nCnt[_FP] + m_nCnt[_TN]); 
}


int32_t PNDetTable::PredNegativeCnt() 
{
	return (m_nCnt[_FN] + m_nCnt[_TN]); 
}


double PNDetTable::TPR()
{
	if(m_nCnt[_TP] > 0)
		return (double)(m_nCnt[_TP]) / (double)(m_nCnt[_TP] + m_nCnt[_FN]); 
	return 0.0; 
}


double PNDetTable::FPR()
{
	if(m_nCnt[_FP] > 0)
		return (double)(m_nCnt[_FP]) / (double)(m_nCnt[_FP] + m_nCnt[_TN]); 
	return 0.0; 
}	


double PNDetTable::TNR()
{
	if(m_nCnt[_TN] > 0)
		return (double)(m_nCnt[_TN]) / (double)(m_nCnt[_FP] + m_nCnt[_TN]); 
	return 0.0; 
}


double PNDetTable::FNR()
{
	if(m_nCnt[_FN] > 0)
		return (double)(m_nCnt[_FN]) / (double)(m_nCnt[_TP] + m_nCnt[_FN]); 
	return 0.0; 
}


double PNDetTable::PosPrecision()
{
	if(m_nCnt[_TP] > 0)
		return (double)(m_nCnt[_TP]) / (double)(m_nCnt[_TP] + m_nCnt[_FP]); 
	return 0.0; 

}


double PNDetTable::NegPrecision()
{
	if(m_nCnt[_TN] > 0)
		return (double)(m_nCnt[_TN]) / (double)(m_nCnt[_FN] + m_nCnt[_TN]); 
	return 0.0; 
}


double PNDetTable::PosRecall()
{
	return TPR(); 
}


double PNDetTable::NegRecall()
{
	return TNR(); 
}


double PNDetTable::PosF1()
{
	double pr = PosPrecision(); 
	double rc = PosRecall(); 
	if(pr == 0.0 || rc == 0.0)
		return 0.0; 
	return 2.0 * pr * rc / (pr + rc); 
}


double PNDetTable::NegF1()
{
	double pr = NegPrecision(); 
	double rc = NegRecall(); 
	if(pr == 0.0 || rc == 0.0)
		return 0.0; 
	return 2.0 * pr * rc / (pr + rc); 
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
// Class RocAnalyzer
/////////////////////////////////////////////////////////////////////////////////////////////////////


RocAnalyzer::RocAnalyzer(const double dScoreFrom, const double dScoreTo) : 
	m_dScoreFrom(dScoreFrom), m_dScoreTo(dScoreTo), m_nPosCnt(0), m_nNegCnt(0)
{
}


RocAnalyzer::~RocAnalyzer()
{
}


void RocAnalyzer::Clear()
{
	m_vtrLabelScore.clear(); 
	m_nPosCnt = 0; 
	m_nNegCnt = 0;
}


void RocAnalyzer::Insert(const ELabelType realLabel, const double dScore) 
{
	if(dScore < m_dScoreFrom)
		m_vtrLabelScore.push_back(pair<ELabelType,double>(realLabel, m_dScoreFrom)); 
	else if(dScore > m_dScoreTo)
		m_vtrLabelScore.push_back(pair<ELabelType,double>(realLabel, m_dScoreTo)); 
	else
		m_vtrLabelScore.push_back(pair<ELabelType,double>(realLabel, dScore)); 

	if(realLabel == _POSITIVE)
		m_nPosCnt++; 
	if(realLabel == _NEGATIVE)
		m_nNegCnt++; 
}


double RocAnalyzer::Auc()
{
	sort(m_vtrLabelScore.begin(), m_vtrLabelScore.end(), Cmp); 
	double auc = 0.0; 
	int32_t positive_cnt_bef = 0;	
	int32_t negative_cnt_bef = 0;	
	int32_t offset = 0; 
	int32_t sub_offset = 0; 

	while(offset < m_nPosCnt + m_nNegCnt - 1)
	{
		if(m_vtrLabelScore[offset].first == _POSITIVE)
		{
			auc += (double)(m_nNegCnt - negative_cnt_bef); 
			sub_offset = offset+1; 
			while(sub_offset < m_nPosCnt + m_nNegCnt)
			{
				if(m_vtrLabelScore[sub_offset].second == m_vtrLabelScore[offset].second)
				{
					if(m_vtrLabelScore[sub_offset].first == _NEGATIVE)
						auc -= 0.5; 
				}
				else
					break; 
				sub_offset++; 
			}
			positive_cnt_bef++; 
		}
		else if(m_vtrLabelScore[offset].first == _NEGATIVE)
		{
			negative_cnt_bef++; 
		}
		offset++; 
	}

	auc /= ((double)m_nPosCnt * (double)m_nNegCnt); 
	return auc; 
}


void RocAnalyzer::RocCurve(vector<pair<double, PNDetTable> >& curve, const double dStep) 
{
	curve.clear(); 
	PNDetTable pn_det_table; 
	double th = m_dScoreFrom; 

	while(th <= m_dScoreTo)
	{
		pn_det_table.Clear(); 
		for(int32_t i = 0; i < (int32_t)m_vtrLabelScore.size(); i++) 
		{
			if(m_vtrLabelScore[i].first == _POSITIVE && 
					m_vtrLabelScore[i].second > th)
			{ // tp, true positive
				pn_det_table.m_nCnt[_TP] += 1;  
			}
			else if(m_vtrLabelScore[i].first == _POSITIVE && 
					m_vtrLabelScore[i].second <= th)
			{ // fn, false negative 
				pn_det_table.m_nCnt[_FN] += 1;  
			}
			else if(m_vtrLabelScore[i].first == _NEGATIVE && 
					m_vtrLabelScore[i].second > th)
			{ // fp, false positive
				pn_det_table.m_nCnt[_FP] += 1;  
			}
			else if(m_vtrLabelScore[i].first == _NEGATIVE && 
					m_vtrLabelScore[i].second <= th)
			{ // tn, true negative 
				pn_det_table.m_nCnt[_TN] += 1;  
			}
		}
		curve.push_back(pair<double, PNDetTable>(th, pn_det_table)); 
		th += dStep; 
	}
}


int32_t RocAnalyzer::Count(const ELabelType eType)
{
	if(eType == _POSITIVE)
		return m_nPosCnt; 
	else
		return m_nNegCnt; 
}


bool RocAnalyzer::Cmp(pair<ELabelType,double> a, pair<ELabelType, double> b)
{
	if(a.second > b.second)
		return true; 
	else if(a.second == b.second)
	{
		if(a.first == _POSITIVE && b.first == _NEGATIVE)
			return true; 
	}
	return false; 
}



