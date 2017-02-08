#include "Matrix.h"
using namespace metis_nn;
#include "Random.h"
#include "StringArray.h"
using namespace metis_uti;
#include <stdio.h>
#include <string.h>


//////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction 

Matrix::Matrix() : m_nRows(0), m_nCols(0), m_data(NULL)
{
}


Matrix::Matrix(const int32_t nRows, const int32_t nCols) : m_nRows(nRows), m_nCols(nCols), m_data(NULL)
{
	m_data = new double*[nRows];
	for(int32_t i = 0; i < nRows; i++) 
		m_data[i] = new double[nCols]; 
}


Matrix::~Matrix()
{
	Release(); 
}


//////////////////////////////////////////////////////////////////////////////////////////
// Operations 

void Matrix::Create(const int32_t nRows, const int32_t nCols)
{
	if(m_data)
	{
		for(int32_t i = 0; i < m_nRows; i++) 
			delete m_data[i]; 
		delete m_data; 
		m_data = NULL; 
	}

	m_nRows = nRows; 
	m_nCols = nCols;
	if(m_nRows > 0 && m_nCols > 0)
	{
		m_data = new double*[nRows];
		for(int32_t i = 0; i < nRows; i++) 
			m_data[i] = new double[nCols]; 
	}
}


void Matrix::Release()
{
	if(m_data)
	{
		for(int32_t i = 0; i < m_nRows; i++) 
			delete m_data[i]; 
		delete m_data; 
		m_data = NULL; 
	}
	m_nRows = 0;	
	m_nCols = 0;	
} 


void Matrix::Init(const double dVal)
{
	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++) 
			m_data[i][j] = dVal; 
	}
}


void Matrix::Init_RandUni(const double left, const double right)
{
	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++) 
			m_data[i][j] = Random::RandUni(left, right); 
	}
}


void Matrix::Init_RandNormal(const double mu, const double sigma)
{
	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++) 
			m_data[i][j] = Random::RandNormal(mu, sigma); 
	}
}


double* Matrix::operator [] (const int32_t nRow)
{
	if(nRow < 0 || nRow >= m_nRows)
		throw "Matrix::[] ERROR: index is out of bounds!"; 
	return m_data[nRow]; 
}


bool Matrix::IsNull()
{
	return (m_nRows == 0 || m_nCols == 0);  
}


int32_t Matrix::Rows()
{
	return m_nRows; 
}


int32_t Matrix::Cols()
{
	return m_nCols; 
}


void Matrix::Sparsification(const double dSpTh)
{
	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++)
		{
			if(m_data[i][j] < dSpTh)
				m_data[i][j] = 0.0; 
		}
	}
}


void Matrix::Print_Matrix(ostream& os, Matrix& mat)
{
	for(int32_t i = 0; i < mat.Rows(); i++) 
	{
		for(int32_t j = 0; j < mat.Cols(); j++) 
		{
			if(j == 0)
				os<<mat[i][j];
			else
				os<<","<<mat[i][j];
		}
		os<<endl; 
	}
}


bool Matrix::Read_Matrix(Matrix& mat, istream& is)
{
	int32_t row = 0;
	string str; 
	while(row < mat.Rows())
	{
		if(is.eof())
			return false; 
		std::getline(is, str); 
		StringArray ar(str.c_str(), ","); 
		if(ar.Count() != mat.Cols())
			return false;
		for(int32_t j = 0; j < mat.Cols(); j++) 
			sscanf(ar.GetString(j).c_str(), "%lf", &(mat[row][j]));  
		row++; 	
	}
	return true; 
}


string Matrix::ToString()
{
	if(IsNull())
		return string("0,0");

	char stmp[64]; 
	sprintf(stmp, "%d,%d", m_nRows, m_nCols);
	string str(stmp); 

	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++) 
		{
			sprintf(stmp, ",%.12g", m_data[i][j]);
			str += stmp; 	
		}
	}
	return str; 
}


bool Matrix::FromString(const char* sStr)
{
	StringArray ar(sStr, ","); 
	if(ar.Count() < 2)
		return false; 
	int32_t r, c; 
	sscanf(ar.GetString(0).c_str(), "%d", &r); 
	sscanf(ar.GetString(1).c_str(), "%d", &c); 
	if(ar.Count() < 2 + r * c)
		return false; 
	Create(r, c);
	for(int32_t i = 0; i < r; i++) 
	{
		for(int32_t j = 0; j < c; j++)
		{
			sscanf(ar.GetString(2 + i * c + j).c_str(), "%lf", &(m_data[i][j])); 
		}
	}

	return true; 
}


bool Matrix::CombineWith(Matrix& mat, const double w0, const double w1)
{
	if(mat.Rows() != m_nRows || mat.Cols() != m_nCols)
		return false; 
	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++) 
			m_data[i][j] = m_data[i][j] * w0 + mat[i][j] * w1;  	
	}
	return true; 
}


void Matrix::NumericMultiWith(const double a)
{
	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++) 
			m_data[i][j] *= a; 
	}
}


int32_t Matrix::ToStream(char* bitStream)
{
	int32_t off = 0; 
	memcpy(bitStream + off, &m_nRows, sizeof(int32_t));
	off += sizeof(int32_t); 
	memcpy(bitStream + off, &m_nCols, sizeof(int32_t));
	off += sizeof(int32_t); 
	for(int32_t i = 0; i < m_nRows; i++)
	{
		memcpy(bitStream + off, m_data[i], m_nCols * sizeof(double));
		off += m_nCols * sizeof(double); 
	}
	return off; 
}


bool Matrix::FromStream(const char* bitStream, const int32_t nLen)
{
	if(nLen < (int32_t)sizeof(int32_t) * 2)
		return false;
	int32_t off = 0;  
	int32_t rows = *((int32_t*)(bitStream + off)); 
	off += sizeof(int32_t); 
	int32_t cols = *((int32_t*)(bitStream + off)); 
	off += sizeof(int32_t); 
	if(nLen < off + rows * cols * (int32_t)sizeof(double))
		return false; 
	Create(rows, cols); 
	for(int32_t i = 0; i < rows; i++)
	{
		memcpy(m_data[i], bitStream + off, sizeof(double) * cols); 
		off += sizeof(double) * cols; 
	}	
	return true; 
}


bool Matrix::Add(Matrix& Y, Matrix& X1, Matrix& X2, const double w1, const double w2)
{
	if(X1.Rows() != X2.Rows() || X1.Cols() != X2.Cols())
		return false; 
	Y.Create(X1.Rows(), X1.Cols());
	for(int32_t i = 0; i < X1.Rows(); i++) 
	{
		for(int32_t j = 0; j < X1.Cols(); j++) 
			Y[i][j] = w1 * X1[i][j] + w2 * X2[i][j]; 	
	}
	return true; 
}


bool Matrix::Minus(Matrix& Y, Matrix& X1, Matrix& X2, const double w1, const double w2)
{
	return Add(Y, X1, X2, w1, 0.0 - w2); 
}


bool Matrix::Multi(Matrix& Y, Matrix& X1, Matrix& X2, const double a)
{
	if(X1.Cols() != X2.Rows())
		return false; 
	Y.Create(X1.Rows(), X2.Cols());
	for(int32_t i = 0; i < X1.Rows(); i++) 
	{
		for(int32_t j = 0; j < X2.Cols(); j++)
		{
			Y[i][j] = 0.0; 
			for(int32_t k = 0; k < X1.Cols(); k++)
				Y[i][j] += X1[i][k] * X2[k][j];	
			Y[i][j] *= a; 
		} 
	}
	return true; 
}


void Matrix::NumericMulti(Matrix& Y, Matrix& X, const double a)
{
	Y.Create(X.Rows(), X.Cols()); 
	for(int32_t i = 0; i < X.Rows(); i++) 
	{
		for(int32_t j = 0; j < X.Cols(); j++) 
			Y[i][j] = a * X[i][j]; 
	}
}


int32_t Matrix::MatsToStream(char* bitStream, Matrix* mats, const int32_t nCnt) 
{
	int32_t off = 0; 
	memcpy(bitStream + off, &nCnt, sizeof(int32_t)); 
	off += sizeof(int32_t); 

	int32_t len; 	
	for(int32_t i = 0; i < nCnt; i++) 
	{
		len = mats[i].ToStream(bitStream + off + sizeof(int32_t)); 
		memcpy(bitStream + off, &len, sizeof(int32_t)); 
		off += sizeof(int32_t) + len;
	}	

	return off; 
}


Matrix* Matrix::MatsFromStream(int32_t& nCnt, const char* bitStream, const int32_t nLen)
{
	int32_t off = 0; 
	if(nLen < off + (int32_t)sizeof(int32_t))
		return NULL; 
	nCnt = *((int32_t*)(bitStream + off)); 
	off += sizeof(int32_t); 

	Matrix* mats = new Matrix[nCnt]; 
	int32_t len; 

	for(int32_t i = 0; i < nCnt; i++) 
	{
		if(nLen < off + (int32_t)sizeof(int32_t))
		{
			delete [] mats; 
			return NULL; 
		}
		len = *((int32_t*)(bitStream + off));
		off += sizeof(int32_t); 
		if(nLen < off + len)
		{
			delete [] mats; 
			return NULL; 
		}
		if(!mats[i].FromStream(bitStream + off, len))	
		{
			delete [] mats; 
			return NULL; 
		}
		off += len; 
	}

	return mats; 
}


int32_t Matrix::MatsStreamSize(Matrix* mats, const int32_t nCnt)
{
	int32_t len = sizeof(int32_t);		// for number of matrices 
	for(int32_t i = 0; i < nCnt; i++)
	{
		len += sizeof(int32_t) * 3;	// for len, rows & columns of the current matrix	
		len += mats[i].Rows() * mats[i].Cols() * sizeof(double); 	// for data of the current matrix
	}
	return len; 
}


bool Matrix::ParseMatrixFromStream(Matrix& mat, const int32_t nIdx, const char* bitStream, const int32_t nLen)
{
	int32_t off = 0; 
	if(nLen < off + (int32_t)sizeof(int32_t))
		return NULL; 
	int32_t cnt = *((int32_t*)(bitStream + off)); 
	off += sizeof(int32_t); 
	if(nIdx < 0 || nIdx >= cnt)
		return false; 	
	
	int32_t len; 	
	for(int32_t i = 0; i < nIdx; i++) 
	{
		if(nLen < off + (int32_t)sizeof(int32_t))
			return false; 
		len = *((int32_t*)(bitStream + off));;
		off += sizeof(int32_t);
		if(nLen < off + len)
			return false; 
		off += len;  
	}
		
	if(nLen < off + (int32_t)sizeof(int32_t))
		return false; 
	len = *((int32_t*)(bitStream + off));;
	off += sizeof(int32_t);
	if(nLen < off + len)
		return false; 
	return mat.FromStream(bitStream + off, len); 
}


