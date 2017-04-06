#include "Pattern.h"
using namespace metis_nn;
#include "StringArray.h"
using namespace metis_uti;
#include <fstream>
#include <iostream>
using namespace std; 
#include <string.h>
#include <math.h>
#include <stdio.h>


//////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction 

Pattern::Pattern() : m_nXCnt(0), m_nYCnt(0), m_x(NULL), m_y(NULL)
{
}


Pattern::Pattern(const int32_t nXCnt, const int32_t nYCnt) : m_nXCnt(nXCnt), m_nYCnt(nYCnt), m_x(NULL), m_y(NULL)
{
	m_x = new double[nXCnt];
	m_y = new double[nYCnt];
}


Pattern::Pattern(const Pattern& patt)
{
	m_nXCnt = patt.m_nXCnt; 
	m_nYCnt = patt.m_nYCnt; 
	m_x = new double[m_nXCnt];
	memcpy(m_x, patt.m_x, patt.m_nXCnt * sizeof(double)); 	
	m_y = new double[m_nYCnt];
	memcpy(m_y, patt.m_y, patt.m_nYCnt * sizeof(double)); 	
}


Pattern::~Pattern()
{
	if(m_x)
	{
		delete m_x; 
		m_x = NULL; 
	}
	if(m_y)
	{
		delete m_y; 
		m_y = NULL; 
	}
}


//////////////////////////////////////////////////////////////////////////////////////////
// Operations 

Pattern& Pattern::operator = (const Pattern& patt)
{
	m_nXCnt = patt.m_nXCnt; 
	m_nYCnt = patt.m_nYCnt; 
	if(m_x)
		delete m_x; 
	m_x = new double[m_nXCnt];
	memcpy(m_x, patt.m_x, patt.m_nXCnt * sizeof(double)); 	
	if(m_y)
		delete m_y; 
	m_y = new double[m_nYCnt];
	memcpy(m_y, patt.m_y, patt.m_nYCnt * sizeof(double)); 	
	return *this; 
}


string Pattern::ToString()
{
	char stmp[64]; 
	string str; 	
	for(int32_t j = 0; j < m_nYCnt; j++) 
	{
		if(j == 0)
			sprintf(stmp, "%.12g", m_y[j]); 
		else
			sprintf(stmp, ",%.12g", m_y[j]); 
		str += stmp; 
	}
	for(int32_t i = 0; i < m_nXCnt; i++) 
	{
		if(i == 0)
			sprintf(stmp, ";%.12g", m_x[i]); 
		else
			sprintf(stmp, ",%.12g", m_x[i]); 
		str += stmp; 
	}

	return str; 
}


bool Pattern::FromString(const char* sStr, const bool bOutliersCheck) 
{
	StringArray array(sStr, ";"); 
	if(array.Count() != 2)
		return false;

	StringArray arr1(array.GetString(0).c_str(), ","); 
	m_nYCnt = (int32_t)arr1.Count(); 	
	if(m_y)
		delete m_y; 
	m_y = new double[m_nYCnt];
	for(int32_t j = 0; j < m_nYCnt; j++)	
		sscanf(arr1.GetString(j).c_str(), "%lf", &(m_y[j]));

	StringArray arr2(array.GetString(1).c_str(), ","); 
	m_nXCnt = (int32_t)arr2.Count(); 	
	if(m_x)
		delete m_x; 
	m_x = new double[m_nXCnt];
	for(int32_t i = 0; i < m_nXCnt; i++)	
	{
		sscanf(arr2.GetString(i).c_str(), "%lf", &(m_x[i]));
		if(bOutliersCheck)
			m_x[i] = fabs(m_x[i]) < 100.0 ? m_x[i] : 0; 
	}
	return true; 
}


int32_t Pattern::ToStream(char* bitStream)
{
	int32_t off = 0; 
	
	memcpy(bitStream + off, &m_nXCnt, sizeof(int32_t)); 
	off += sizeof(int32_t);
	
	memcpy(bitStream + off, &m_nYCnt, sizeof(int32_t)); 
	off += sizeof(int32_t);

	memcpy(bitStream + off, m_x, m_nXCnt * sizeof(double)); 
	off += m_nXCnt * sizeof(double);

	memcpy(bitStream + off, m_y, m_nYCnt * sizeof(double)); 
	off += m_nYCnt * sizeof(double);

	return off; 
}
	

bool Pattern::FromStream(const char* bitStream, const int32_t nLen)
{
	if(!bitStream)
		return false; 
	if(nLen < (int32_t)(sizeof(int32_t) * 2))
		return false; 

	int32_t off = 0; 
	m_nXCnt = *((int32_t*)(bitStream + off)); 
	off += sizeof(int32_t); 
	m_nYCnt = *((int32_t*)(bitStream + off)); 
	off += sizeof(int32_t); 

	if(nLen < int32_t(sizeof(int32_t) * 2 + (m_nXCnt + m_nYCnt) * sizeof(double)))
		return false; 

	if(m_x)
		delete m_x; 
	m_x = new double[m_nXCnt];
	memcpy(m_x, bitStream + off, m_nXCnt * sizeof(double)); 
	off += m_nXCnt * sizeof(double); 

	if(m_y)
		delete m_y; 
	m_y = new double[m_nYCnt];
	memcpy(m_y, bitStream + off, m_nYCnt * sizeof(double)); 
	off += m_nYCnt * sizeof(double); 

	return true; 
}


bool Pattern::LoadPartterns(vector<Pattern*>& vtrPatts, const char* sFile, const bool bSkipHeader)
{
	ifstream ifs(sFile);
	if(!ifs.is_open())
		return false; 

	for(size_t i = 0; i < vtrPatts.size(); i++)
		delete vtrPatts[i]; 
	vtrPatts.clear(); 

	string str; 
	Pattern* ppatt = NULL; 

	if(bSkipHeader)
		std::getline(ifs, str); 
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
	return true; 
}


string Pattern::ArrayToString(const double* x, const int32_t len)
{
	char stmp[64]; 
	string str; 
	for(int32_t i = 0; i < len; i++) 
	{
		if(i == 0)
			sprintf(stmp, "%.12g", x[i]);
		else
			sprintf(stmp, ",%.12g", x[i]);
		str += stmp; 
	}
	return str; 
}


int32_t Pattern::ArrayFromString(double* x, const int32_t len, const char* sStr)
{
	StringArray ar(sStr, ",");
	int32_t idx = 0; 
	while(idx < ar.Count() && idx < len)
	{
		sscanf(ar.GetString(idx).c_str(), "%lf", &x[idx]); 	
		idx++; 
	}
	return idx; 
}


int32_t Pattern::MaxOff(const double* x, const int32_t len)
{
	int32_t max_off = 0; 
	for(int32_t i = 1; i < len; i++) 
	{
		if(x[i] > x[max_off])
			max_off = i; 
	}
	return max_off; 
}


double Pattern::Error(const double* y1, const double* y2, const int32_t len)
{
	double error = 0.0; 
	for(int32_t i = 0; i < len; i++) 
		error += (y1[i] - y2[i]) * (y1[i] - y2[i]); 
	return error / (double)len; 
}


void Pattern::Print_Array(ostream& os, const double* x, const int32_t len)
{
	for(int32_t i = 0; i < len; i++) 
	{
		if(i == 0)
			os<<x[i]; 
		else
			os<<","<<x[i]; 
	}
	os<<endl; 
}


bool Pattern::Read_Array(double* x, const int32_t len, istream& is)
{
	string str; 
	std::getline(is, str); 
	StringArray ar(str.c_str(), ","); 
	if(ar.Count() != len)
		return false; 
	for(int32_t i = 0; i < len; i++) 
		sscanf(ar.GetString(i).c_str(), "%lf", &(x[i])); 
	return true; 
}



