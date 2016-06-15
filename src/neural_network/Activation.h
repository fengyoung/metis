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

	// RelU & its derivation
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

