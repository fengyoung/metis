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
		case _ACT_RELU6:
			y = ReLU6(x);
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
		case _ACT_RELU6:
			d = DReLU6(y);
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


double Activation::ReLU6(const double x)
{
	return min(max(0.0, x), 6.0); 
}


double Activation::DReLU6(const double y)
{
	return (y > 0.0 && y < 6.0) ? 1.0 : 0.0; 
}


double Activation::Softmax(const int32_t k, const double* x, const int32_t len)
{
	double sum = 0.0;
	double yk = 0.0;
	for(int32_t i = 0; i < len; i++) 
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


double Activation::DActRegula(const double v, const ERegula eRegula, const double alpha)
{
	if(eRegula == _REGULA_L1)
	{
		if(v > 0.0)
			return 1.0 * alpha; 
		else if(v < 0.0)
			return -1.0 * alpha; 
		else
			return 0.0;
	}
	else if(eRegula == _REGULA_L2)
		return v * alpha; 
	else	
		return 0.0; 
}


double Activation::Loss_Quadratic(const double y_pred, const double y)
{
	return (y_pred - y) * (y_pred - y) / 2.0; 
}


double Activation::Loss_CrossEntropy(const double y_pred, const double y)
{
	double p = 1e-8; 
	if(y_pred <= 0.0)
	{
		if(y <= 0.0)
			return 0.0; 
		else
			p = 1e-8;
	}
	else if(y_pred >= 1.0)
	{
		if(y >= 1.0)
			return 0.0; 
		else
			p = 1.0 - 1e-8; 
	}
	else
		p = y_pred; 	
	return 0.0 - (y * log(p) + (1.0 - y) * log(1.0 - p)); 	
}


double Activation::Loss_LogLikelihood(const double* y_pred, const double* y, const int32_t y_len)
{
	double p = 1e-8; 
	for(int32_t j = 0; j < y_len; j++) 
	{
		if(y[j] >= 1.0)
		{
			if(y_pred[j] <= 0.0)
				p = 1e-8; 
			else if(y_pred[j] >= 1.0)
				p = 1.0; 
			else
				p = y_pred[j]; 
		}
	}
	return 0.0 - log(p); 
}


