#include "NNAssi.h"
using namespace metis_nn; 
#include <string.h>



//////////////////////////////////////////////////////////////////
// class NNAssi

NNAssi::NNAssi(const EOptim eOptim) : m_eOptim(eOptim), m_ao(NULL), m_do(NULL)
{
}


NNAssi::~NNAssi()
{
	if(m_ao)
	{
		delete m_ao; 
		m_ao = NULL;
	} 
	if(m_do)
	{
		delete m_do; 
		m_do = NULL;
	} 
}


NNAssi* NNAssi::New(const EOptim eOptim, const int32_t nInput, const int32_t nOutput, const bool bForTrain)
{
	NNAssi* p = NULL; 
	switch(eOptim)
	{
		case _OPTIM_SGD:
			p = new NNAssi_SGD();
			break; 
		case _OPTIM_MOMENTUM:
			p = new NNAssi_Momentum(); 
			break; 
		case _OPTIM_NAG:
			p = new NNAssi_NAG(); 
			break; 
		case _OPTIM_ADAGRAD:
			p = new NNAssi_AdaGrad(); 
			break; 
		case _OPTIM_RMSPROP:
			p = new NNAssi_RMSprop(); 
			break; 
		case _OPTIM_ADADELTA:
			p = new NNAssi_AdaDelta(); 
			break; 
		case _OPTIM_ADAM:
			p = new NNAssi_Adam(); 
			break; 
		default:
			break; 
	}
	if(p)
		p->Create(nInput, nOutput, bForTrain); 
	return p; 
}


//////////////////////////////////////////////////////////////////
// class NNAssi_SGD

NNAssi_SGD::NNAssi_SGD() : NNAssi(_OPTIM_SGD)
{
}


NNAssi_SGD::~NNAssi_SGD()
{
	Release(); 
}


void NNAssi_SGD::Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain)
{
	Release(); 
	m_ao = new double[nOutput]; 
	if(bForTrain)
	{
		m_do = new double[nOutput]; 
		memset(m_do, 0, sizeof(double) * nOutput); 	
		m_g.Create(nInput + 1, nOutput); 
		m_g.Init(0.0); 
	}
}


void NNAssi_SGD::Release()
{
	if(m_ao)
	{
		delete m_ao; 
		m_ao = NULL; 
	}
	if(m_do)
	{
		delete m_do; 
		m_do = NULL; 
	}
	m_g.Release(); 
}


//////////////////////////////////////////////////////////////////
// class NNAssi_Momentum

NNAssi_Momentum::NNAssi_Momentum() : NNAssi(_OPTIM_MOMENTUM)
{
}


NNAssi_Momentum::~NNAssi_Momentum()
{
	Release(); 
} 


void NNAssi_Momentum::Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain)
{
	Release(); 
	m_ao = new double[nOutput]; 
	if(bForTrain)
	{
		m_do = new double[nOutput]; 
		memset(m_do, 0, sizeof(double) * nOutput); 	
		m_g.Create(nInput + 1, nOutput); 
		m_g.Init(0.0); 
		m_v_prev.Create(nInput + 1, nOutput); 
		m_v_prev.Init(0.0); 
	}
}


void NNAssi_Momentum::Release()
{
	if(m_ao)
	{
		delete m_ao; 
		m_ao = NULL; 
	}
	if(m_do)
	{
		delete m_do; 
		m_do = NULL; 
	}
	m_g.Release(); 
	m_v_prev.Release(); 
}



//////////////////////////////////////////////////////////////////
// class NNAssi_NAG

NNAssi_NAG::NNAssi_NAG() : NNAssi(_OPTIM_NAG)
{
}


NNAssi_NAG::~NNAssi_NAG()
{
	Release(); 
} 


void NNAssi_NAG::Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain)
{
	Release(); 
	m_ao = new double[nOutput]; 
	if(bForTrain)
	{
		m_do = new double[nOutput]; 
		memset(m_do, 0, sizeof(double) * nOutput); 	
		m_g.Create(nInput + 1, nOutput); 
		m_g.Init(0.0); 
		m_v_prev.Create(nInput + 1, nOutput); 
		m_v_prev.Init(0.0); 
	}
}


void NNAssi_NAG::Release()
{
	if(m_ao)
	{
		delete m_ao; 
		m_ao = NULL; 
	}
	if(m_do)
	{
		delete m_do; 
		m_do = NULL; 
	}
	m_g.Release(); 
	m_v_prev.Release(); 
}


//////////////////////////////////////////////////////////////////
// class NNAssi_AdaGrad

NNAssi_AdaGrad::NNAssi_AdaGrad() : NNAssi(_OPTIM_ADAGRAD)
{
}


NNAssi_AdaGrad::~NNAssi_AdaGrad()
{
	Release(); 
}	


void NNAssi_AdaGrad::Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain)
{
	Release(); 
	m_ao = new double[nOutput]; 
	if(bForTrain)
	{
		m_do = new double[nOutput]; 
		memset(m_do, 0, sizeof(double) * nOutput); 	
		m_g.Create(nInput + 1, nOutput); 
		m_g.Init(0.0); 
		m_g2_acc.Create(nInput + 1, nOutput); 
		m_g2_acc.Init(0.0); 
	}
}


void NNAssi_AdaGrad::Release()
{
	if(m_ao)
	{
		delete m_ao; 
		m_ao = NULL; 
	}
	if(m_do)
	{
		delete m_do; 
		m_do = NULL; 
	}
	m_g.Release(); 
	m_g2_acc.Release(); 
}


//////////////////////////////////////////////////////////////////
// class NNAssi_RMSprop

NNAssi_RMSprop::NNAssi_RMSprop() : NNAssi(_OPTIM_RMSPROP)
{
}


NNAssi_RMSprop::~NNAssi_RMSprop()
{
	Release(); 
}


void NNAssi_RMSprop::Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain)
{
	Release(); 
	m_ao = new double[nOutput]; 
	if(bForTrain)
	{
		m_do = new double[nOutput]; 
		memset(m_do, 0, sizeof(double) * nOutput); 	
		m_g.Create(nInput + 1, nOutput); 
		m_g.Init(0.0); 
		m_g2_mavg.Create(nInput + 1, nOutput); 
		m_g2_mavg.Init(0.0); 
	}
}


void NNAssi_RMSprop::Release()
{
	if(m_ao)
	{
		delete m_ao; 
		m_ao = NULL; 
	}
	if(m_do)
	{
		delete m_do; 
		m_do = NULL; 
	}
	m_g.Release(); 
	m_g2_mavg.Release(); 
}


//////////////////////////////////////////////////////////////////
// class NNAssi_AdaDelta

NNAssi_AdaDelta::NNAssi_AdaDelta() : NNAssi(_OPTIM_ADADELTA)
{
}


NNAssi_AdaDelta::~NNAssi_AdaDelta()
{
	Release(); 
}


void NNAssi_AdaDelta::Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain)
{
	Release(); 
	m_ao = new double[nOutput]; 
	if(bForTrain)
	{
		m_do = new double[nOutput]; 
		memset(m_do, 0, sizeof(double) * nOutput); 	
		m_g.Create(nInput + 1, nOutput); 
		m_g.Init(0.0); 
		m_v2_mavg.Create(nInput + 1, nOutput); 
		m_v2_mavg.Init(0.0); 
		m_g2_mavg.Create(nInput + 1, nOutput); 
		m_g2_mavg.Init(0.0); 
	}
}


void NNAssi_AdaDelta::Release()
{
	if(m_ao)
	{
		delete m_ao; 
		m_ao = NULL; 
	}
	if(m_do)
	{
		delete m_do; 
		m_do = NULL; 
	}
	m_g.Release(); 
	m_v2_mavg.Release(); 
	m_g2_mavg.Release(); 
}


//////////////////////////////////////////////////////////////////
// class NNAssi_Adam

NNAssi_Adam::NNAssi_Adam() : NNAssi(_OPTIM_ADAM)
{
}


NNAssi_Adam::~NNAssi_Adam()
{
	Release(); 
}


void NNAssi_Adam::Create(const int32_t nInput, const int32_t nOutput, const bool bForTrain)
{
	Release(); 
	m_ao = new double[nOutput]; 
	if(bForTrain)
	{
		m_do = new double[nOutput]; 
		memset(m_do, 0, sizeof(double) * nOutput); 	
		m_g.Create(nInput + 1, nOutput); 
		m_g.Init(0.0); 
		m_g_mavg.Create(nInput + 1, nOutput); 
		m_g_mavg.Init(0.0); 
		m_g2_mavg.Create(nInput + 1, nOutput); 
		m_g2_mavg.Init(0.0); 
	}
}


void NNAssi_Adam::Release()
{
	if(m_ao)
	{
		delete m_ao; 
		m_ao = NULL; 
	}
	if(m_do)
	{
		delete m_do; 
		m_do = NULL; 
	}
	m_g.Release(); 
	m_g_mavg.Release(); 
	m_g2_mavg.Release(); 
}


