// Matrix.h
//
// Definition of class Matrix
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v1.0 2016-03-14
//

#ifndef _METIS_NEURAL_NETWORK_MATRIX_H 
#define _METIS_NEURAL_NETWORK_MATRIX_H 

#include <string>
#include <vector>
#include <iostream>
using namespace std;
#include <stdint.h>
#include <sys/time.h>


namespace metis_nn
{

// CLASS
//	Matrix - definition of matrix 
class Matrix
{
public:
	// Construction & Destruction
	Matrix(); 
	Matrix(const int32_t nRows, const int32_t nCols);
	virtual ~Matrix();

	// NAME
	//	Create - create the current matrix, allocate memory
	// 
	// DESCRIPTION
	//	nRows - number of rows
	//	nCols - numer of columes
	void Create(const int32_t nRows, const int32_t nCols); 

	// NAME
	//	Release - release current matrix
	void Release(); 

	// NAME
	//	Init - initialize the matrix, set elements as the same value 
	//	Init_RandUni - initialize the matrix, set elements based on uniform distribution 
	//	Init_RandNormal - initialize the matrix, set elements based on normal distribution
	// 
	// DESCRIPTION
	//	dVal: the initial value
	//	left, right: the minimal and maximal values of the uniform distribution
	//	mu, sigma - mean and standard deviation of the normal distribution
	void Init(const double dVal); 
	void Init_RandUni(const double left, const double right); 
	void Init_RandNormal(const double mu, const double sigma); 

	// Reload the subscrip
	double* operator [] (const int32_t row);

	// NAME
	//	IsNull - determine whether the matrix is null
	//
	// RETURN 
	//	true or false
	bool IsNull();
	
	// NAME 
	//	Rows / Cols - get the number of rows or columes 
	// 
	// RETURN
	//	rows or columes of current matrix
	int32_t Rows(); 
	int32_t Cols(); 

	// NAME
	//	Sparsification - sparsificate the matrix
	//
	// DESCRIPTION
	//	dSpTh: value threshold for sparsification
	void Sparsification(const double dSpTh = 0.000000000001); 

	// NAME
	//	Print_Matrix - print a matrix object to stream
	// 
	// DESCRIPTION
	//	os: output stream
	//	mat: matrix
	static void Print_Matrix(ostream& os, Matrix& mat);
	
	// NAME	
	//	Read_Matrix - read a matrix object from stream
	//
	// DESCRIPTION
	//	mat: matrix
	//	is: input stream
	//
	// RETURN
	//	true for success, false for some errors
	static bool Read_Matrix(Matrix& mat, istream& is);

	// NAME
	//	CombineWith - combine with other matrix. M = M * w0 + M1 * w1 
	//
	// DESCRIPTION
	//  mat: matrix which is used for combination
	//  w0: weight of current matrix
	//  w1: weight of mat
	//
	// RETURN
	//	true for success, false for some errors
	bool CombineWith(Matrix& mat, const double w0 = 1.0, const double w1 = 1.0);
	
	// NAME
	//	NumericMultiWith - multi with a number, M = M * a
	//
	// DESCRIPTION
	// 	a: the multiplicand
	//
	void NumericMultiWith(const double a); 

	// NAME
	//	ToString - trasform current matrix to string format
	//	FromString - parse string to construct current matrix
	//
	// DESCRIPTION
	//	sStr: matrix in string
	//
	// RETURN
	//	the matrix in string
	//	true for success, false for some errors
	string ToString();
	bool FromString(const char* sStr); 

	// NAME
	//	ToStream - trasform current matrix to bit-stream format
	//	FromStream - parse bit-stream to construct current matrix
	//
	// DESCRIPTION
	//	bitStream: matrix in bit-stream
	//	nLen: length of matrix stream
	//
	// RETURN
	//	length of matrix stream
	//	true for success, false for some errors
	int32_t ToStream(char* bitStream); 
	bool FromStream(const char* bitStream, const int32_t nLen);  

	// NAME
	// 	Add - matrix add another one, Y = w1 * X1 + w2 * X2
	// 	Minus - matrix minus another one, Y = w1 * X1 - w2 * X2
	// 	Multi - matrix multiplied by another one with multiple number, Y = a * X1 * X2 
	// 	NumericMulti - matrix multiplied by a numerical value, Y = a * X
	static bool Add(Matrix& Y, Matrix& X1, Matrix& X2, const double w1 = 1.0, const double w2 = 1.0);  
	static bool Minus(Matrix& Y, Matrix& X1, Matrix& X2, const double w1 = 1.0, const double w2 = 1.0);  
	static bool Multi(Matrix& Y, Matrix& X1, Matrix& X2, const double a = 1.0);
	static void NumericMulti(Matrix& Y, Matrix& X, const double a); 

	// NAME
	// 	MatsToStream - matrices to bit-stream
	// 	MatsFromStream - parse bit-stream to construct matrices
	// 	MatsStreamSize - estimate the size of matrices stream 
	// 	ParseMatrixFromStream - parse one matrix from matrices stream 
	static int32_t MatsToStream(char* bitStream, Matrix* mats, const int32_t nCnt);  
	static Matrix* MatsFromStream(int32_t& nCnt, const char* bitStream, const int32_t nLen); 
	static int32_t MatsStreamSize(Matrix* mats, const int32_t nCnt); 
	static bool ParseMatrixFromStream(Matrix& mat, const int32_t nIdx, const char* bitStream, const int32_t nLen); 

private: 
	int32_t m_nRows;	// number of rows 
	int32_t m_nCols;	// number of columes
	double** m_data;	// 2-d array, which is used for storing element values
};

}

#endif /* _METIS_NEURAL_NETWORK_MATRIX_H */


