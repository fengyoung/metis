// Activation.h
//
// Definition of activation functions and their derivations
//
// There are 5 types of activations: Linear, Sigmoid, Tanh, ReLU and Softmax
// +----------+-------------------------------------------------+-----------------------+
// | Linear   | y(x) = x                                        |  y'(x) = 1            | 
// +----------+-------------------------------------------------+-----------------------+
// | Sigmoid  | y(x) = 1 / [1 + exp(-x)]                        |  y'(x) = y * (1 - y)  |
// +----------+-------------------------------------------------+-----------------------+
// | Tanh     | y(x) = [exp(x) - exp(-x)] / [exp(x) + exp(-x)]  |  y'(x) = 1 - y^2      |
// +----------+-------------------------------------------------+-----------------------+
// | ReLU     | y(x) = MAX(0, x)                                |  y'(x) = 0 or 1       |
// +----------+-------------------------------------------------+-----------------------+
// | Softmax  | y(xi) = exp(xi) / SUM(exp(xj))                  |  y'(x) = y * (1 - y)  |
// +----------+-------------------------------------------------+-----------------------+
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v1.0 2016-03-14
//

#ifndef _METIS_NN_MATRIX_ACTIVATION_H 
#define _METIS_NN_MATRIX_ACTIVATION_H 

#include <map>
#include <string>
using namespace std;
#include <stdint.h>
#include "Matrix.h"
#include "TypeDefs.h"


namespace metis_nn
{

// CLASS
//	Activation - definition of activation functions
// 
// DESCRIPTION
//	Activation function is the nonlinear transformer of neruon.
//	The activation function must be non-constant, bounded, monotone-increasing and continuous
//
class Activation
{
private:
	// Construction & Destruction
	Activation(); 
	virtual ~Activation(); 
	
public:
	// NAME
	//	Activate - calculate the value of activation function 
	//	DActivate - calculate the value of derivation of the activation function  
	//
	// DESCRIPTION
	//	x: value of activation potential  
	//	y: value of activation function
	//	eActType: type of activation
	//
	// RETURN
	//	The value of activation function or the derivation
	static double Activate(const double x, const EActType eActType);
	static double DActivate(const double y, const EActType eActType);

	// Linear & its derivation
	static double Linear(const double x); 
	static double DLinear(); 

	// Sigmoid & its derivation
	static double Sigmoid(const double x);
	static double DSigmoid(const double y); 

	// Tanh & its derivation
	static double Tanh(const double x); 
	static double DTanh(const double y); 

	// ReLU & its derivation
	static double ReLU(const double x); 
	static double DReLU(const double y); 

	// Softmax & its derivation
	static double Softmax(const double k, const double* x, const uint32_t len); 
	static double DSoftmax(const double y); 

	// NAME
	//	DActRegula - get derivation of the regularization term
	// 
	// DESCRIPTION
	//	v: element value of transform matrix
	//	eRegula: regularization type
	//	lambda: tiny constant
	//
	// RETRUN
	//	The value of derivation
	static double DActRegula(const double v, const ERegula eRegula, const double lambda = 0.01); 

	// NAME
	//	InitTransformMatrix - initialize the transform matrix based on the type of activation
	//
	// DESCRIPTION
	//	w: transform matrix
	//	eActType: type of activation
	static void InitTransformMatrix(Matrix& w, const EActType eActType);
};

}

#endif /* _METIS_NN_MATRIX_ACTIVATION_H */

