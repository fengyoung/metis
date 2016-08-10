// Pattern.h
//
// Definition of class Pattern, which is used for stroing training samples
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v1.0 2016-03-14
//

#ifndef _METIS_NN_PATTERN_H 
#define _METIS_NN_PATTERN_H 

#include <string>
#include <vector>
#include <iostream>
using namespace std; 
#include <stdint.h>


namespace metis_nn
{

// CLASS
//	Pattern - definition of sample data (pattern) 
// 
// DESCRIPTION
//	Pattern consists independent variables (Y) and dependent variables (X)
//	Pattern is used for model training and varification
//
class Pattern
{
public:
	// Construction & Destruction
	//
	// DESCRIPTION
	//	nXCnt / nYCnt: size of feature vector and label arrays
	Pattern(); 
	Pattern(const int32_t nXCnt, const int32_t nYCnt); 
	Pattern(const Pattern& patt); 
	virtual ~Pattern(); 

	// Reload assignment 
	Pattern& operator = (const Pattern& patt); 

	// NAME
	//	ToString - transform current pattern to string in fomat "y1,y2,...,ym;x1,x2,x3,...,xn"
	//
	// RETRUN
	//	pattern string
	string ToString();  
	
	// NAME	
	//	FromString - parse string and create current pattern
	//
	// DESCRITPION
	//	sStr: pattern in string format
	//
	// RETURN
	//	true for success, false for some errors
	bool FromString(const char* sStr);  

	// NAME
	//	ToStream - transform current pattern to bit-stream 
	//
	// DESCRIPTION
	//	bitStream - output bit-stream which stores pattern data
	//
	// RETURN
	//	length of the bit-stream
	int32_t ToStream(char* bitStream);

	// NAME
	//	FromStream - parse bit-stream and create current pattern 
	//
	// DESCRIPTION
	//	bitStream - input bit-stream which stores pattern data
	//	nLen - length of bit-stream
	//
	// RETURN
	//	true for success, false for some errors
	bool FromStream(const char* bitStream, const int32_t nLen); 

	// NAME
	//	LoadPartterns - load partterns from data file
	// 
	// DESCRIPTION
	//	vtrPatts: out parameter, the list of patterns
	//	sFile: data file, each row is one pattern string
	//	bSkipHeader: true for skipping header line, false for none
	//
	// RETURN
	//	true for success, false for some errors
	static bool LoadPartterns(vector<Pattern*>& vtrPatts, const char* sFile, const bool bSkipHeader = false); 

	// NAME
	//	ArrayToString - transform a value array to string in format "x1,x2,...,xn"
	//
	// DESCRIPTION
	//	x: the value array
	//	len: length of the array
	//
	// RETURN
	//	value array string
	static string ArrayToString(const double* x, const int32_t len);

	// NAME
	//	MaxOff - get the offset of the maximal value in the array
	//
	// DESCRIPTION
	//	x: the value array
	//	len: length of the array
	//
	// RETURN 
	//	The offset
	static int32_t MaxOff(const double* x, const int32_t len);

	// NAME
	//	Error - calculate the difference between tow value arrays in the same size
	// 
	// DESCRIPTION
	//	error = (y1 - y2)^T * (y1 - y2)
	//
	//	y1, y2: tow value arrays
	//	len: size of value array
	//
	// RETURN
	//	The difference value
	static double Error(const double* y1, const double* y2, const int32_t len); 

	// NAME
	//	Print_Array - print an array to stream
	//
	// DESCRIPTION
	//	os: output stream 
	//	x: array
	//	len: length of the array
	static void Print_Array(ostream& os, const double* x, const int32_t len); 

	// NAME
	//  Read_Array - read an array from stream
	//
	// DESCRIPTION
	//	x: array
	//	len: length of the array
	//	is: input stream 
	// 
	// RETURN
	//	true for success, false for some error
	static bool Read_Array(double* x, const int32_t len, istream& is); 

public:
	int32_t m_nXCnt;	// number of independent variables
	int32_t m_nYCnt;	// number of dependent variables
	double* m_x;		// independent variables
	double* m_y;		// dependent variables 
};

}

#endif /* _METIS_NN_PATTERN_H */

