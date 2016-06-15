#ifndef _METIS_NN_MATRIX_H 
#define _METIS_NN_MATRIX_H 

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

	// Determine whether the matrix is null
	bool IsNull(); 

	// Get the number of rows or columes 
	int32_t Rows(); 
	int32_t Cols(); 

	// Sparsificate the matrix
	void Sparsification(const double dSpTh = 0.000000000001); 

	// Print a matrix object to stream
	static void Print_Matrix(ostream& os, Matrix& mat);
	// Read a matrix object from stream
	static bool Read_Matrix(Matrix& mat, istream& is);

	bool CombineWith(Matrix& mat, const double w0 = 1.0, const double w1 = 1.0);

	string ToString(); 
	bool FromString(const char* sStr); 

private: 
	int32_t m_nRows;	// number of rows 
	int32_t m_nCols;	// number of columes
	double** m_data;	// 2-d array, which is used for storing element values
};

}

#endif /* _METIS_NN_MATRIX_H */

