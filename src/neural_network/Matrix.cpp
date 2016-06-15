#include "Matrix.h"
using namespace metis_nn; 
#include "Random.h"
#include "StringArray.h"
using namespace metis_uti; 
#include <stdio.h>


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
			sprintf(stmp, ",%.6g", m_data[i][j]);
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
			sscanf(ar.GetString(2 + i * c + j).c_str(), "%lf", &(m_data[i][j])); 
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


