#include "Activation.h"
using namespace metis_nn; 
#include <math.h>
#include <string.h>


//////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction 


Activation::Activation() 
{
}


Activation::~Activation() 
{
}


//////////////////////////////////////////////////////////////////////////////////////////
// Operations 

double Activation::Activate(const double x, const EActType eActType)
{
	double y = x; 
	switch(eActType)
	{
		case _ACT_LINEAR: 
			y = Linear(x);
			break; 
		case _ACT_SIGMOID: 
			y = Sigmoid(x);
			break; 
		case _ACT_TANH:
			y = Tanh(x);
			break; 
		case _ACT_RELU:
			y = ReLU(x);
			break; 
		case _ACT_SOFTMAX:
			y = exp(x);
			break; 
		default: 
			break; 
	}
	return y; 
}


double Activation::DActivate(const double y, const EActType eActType)
{
	double d = 0.0; 
	switch(eActType)
	{
		case _ACT_LINEAR: 
			d = DLinear(); 
			break; 
		case _ACT_SIGMOID: 
			d = DSigmoid(y);
			break; 
		case _ACT_TANH:
			d = DTanh(y);
			break; 
		case _ACT_RELU:
			d = DReLU(y);
			break; 
		case _ACT_SOFTMAX:
			d = DSoftmax(y);
			break; 
		default: 
			break; 
	}
	return d; 
}


double Activation::Linear(const double x)
{
	return x;
}


double Activation::DLinear()
{
	return 1.0; 
}


double Activation::Sigmoid(const double x)
{
	return 1.0 / (1.0 + exp(0.0 - x));
}


double Activation::DSigmoid(const double y) 
{
	return y * (1.0 - y);
}


double Activation::Tanh(const double x)
{
	return tanh(x);
}


double Activation::DTanh(const double y)
{
	return (1.0 - y * y); 
}


double Activation::ReLU(const double x)
{
	return max(0.0, x);	
}


double Activation::DReLU(const double y)
{
	return (y > 0.0 ? 1.0 : 0.0); 
}


double Activation::Softmax(const double k, const double* x, const uint32_t len)
{
	double sum = 0.0;
	double yk = 0.0;
	for(uint32_t i = 0; i < len; i++) 
	{
		if(i == k)
		{
			yk = exp(x[i]);	
			sum += yk;  	
		}
		else
			sum += exp(x[i]);	
	}
	return yk / sum;
}


double Activation::DSoftmax(const double y)
{
	return y * (1.0 - y);
}


double Activation::DActRegula(const double v, const ERegula eRegula, const double lambda)
{
	if(eRegula == _REGULA_L1)
	{
		if(v > 0.0)
			return 1.0 * lambda; 
		else if(v < 0.0)
			return -1.0 * lambda; 
		else
			return 0.0;
	}
	else if(eRegula == _REGULA_L2)
		return v * lambda; 
	else	
		return 0.0; 
}


void Activation::InitTransformMatrix(Matrix& w, const EActType eActType)
{
	if(eActType == _ACT_SIGMOID)
		w.Init_RandNormal(0.0, sqrt(1.0 / (double)w.Rows())); 
	else if(eActType == _ACT_TANH)
		w.Init_RandNormal(0.0, sqrt(1.0 / (double)w.Rows())); 
	else if(eActType == _ACT_RELU)
		w.Init_RandNormal(0.0, sqrt(1.0 / (double)w.Rows())); 
	else if(eActType == _ACT_SOFTMAX)
		w.Init_RandNormal(0.0, sqrt(1.0 / (double)w.Rows())); 
	else
		w.Init_RandUni(-4.0 * sqrt(6.0 / (double)(w.Rows() + w.Cols())), 4.0 * sqrt(6.0 / (double)(w.Rows() + w.Cols()))); 
}



