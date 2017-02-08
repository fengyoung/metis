#include "SlaveTrainer.h"
using namespace metis_plat; 
#include <algorithm>
using namespace std; 
#include <string.h>
#include <math.h>


bool SlaveTrainer::m_bCancelFlag = false;  


SlaveTrainer::SlaveTrainer() : m_ws(NULL), m_gs(NULL), m_aos(NULL), m_dos(NULL), m_hl(-1), m_nPattOff(0) 
{
}


SlaveTrainer::~SlaveTrainer()
{
	ReleaseWeight(); 
	ReleasePatts(); 
}


bool SlaveTrainer::PushPatt_inString(const char* sPattStr)
{
	Pattern* ppatt = new Pattern();
	if(!ppatt->FromString(sPattStr))
	{
		delete ppatt; 
		return false; 
	}
	m_vtrPatts.push_back(ppatt); 
	return true; 
}


bool SlaveTrainer::PushPatt_inStream(const char* sPattStream, const int32_t nLen)
{
	Pattern* ppatt = new Pattern(); 
	if(!ppatt->FromStream(sPattStream, nLen))
	{
		delete ppatt; 
		return false; 
	}
	m_vtrPatts.push_back(ppatt); 
	return true; 
}


int32_t SlaveTrainer::PattCnt()
{
	return (int32_t)m_vtrPatts.size(); 
}


PattsInfoT SlaveTrainer::GetPattsInfo()
{
	PattsInfoT patts_info_t;  
	if(m_vtrPatts.empty())
	{
		patts_info_t.x_dims = -1;  
		patts_info_t.y_dims = -1;  
		patts_info_t.patts = 0; 
	}
	else
	{
		patts_info_t.x_dims = m_vtrPatts[0]->m_nXCnt;  
		patts_info_t.y_dims = m_vtrPatts[0]->m_nYCnt;  
		patts_info_t.patts = (int32_t)m_vtrPatts.size(); 
	}
	return patts_info_t;  
}


void SlaveTrainer::PattShuffle()
{
	random_shuffle(m_vtrPatts.begin(), m_vtrPatts.end()); 
}


void SlaveTrainer::ReleasePatts()
{
	for(size_t i = 0; i < m_vtrPatts.size(); i++)
		delete m_vtrPatts[i]; 
	m_vtrPatts.clear(); 
	m_nPattOff = 0; 
}


bool SlaveTrainer::SetWeight_inStream(const char* sWeightStream, const int32_t nLen)
{
	ReleaseWeight(); 
	m_ws = Matrix::MatsFromStream(m_hl, sWeightStream, nLen); 
	if(!m_ws)
	{
		m_hl = -1; 
		return false; 
	}
	m_hl -= 1; 
	m_gs = new Matrix[m_hl+1];
	m_aos = new double*[m_hl+1];
	m_dos = new double*[m_hl+1];

	for(int32_t h = 0; h <= m_hl; h++) 
	{
		m_gs[h].Create(m_ws[h].Rows(), m_ws[h].Cols()); 
		m_gs[h].Init(0.0); 
		m_aos[h] = new double[m_ws[h].Cols()];
		m_dos[h] = new double[m_ws[h].Cols()];
	}	

	return true; 
}

int32_t SlaveTrainer::WeightStreamLen()
{
	if(!m_ws)
		return (int32_t)sizeof(int32_t); 
	int32_t len = (int32_t)sizeof(int32_t);  
	for(int32_t h = 0; h <= m_hl; h++) 
	{
		len += 3 * (int32_t)sizeof(int32_t);  
		len += m_ws[h].Rows() * m_ws[h].Cols() * sizeof(double); 
	}
}


int SlaveTrainer::GetGrad_asStream(char* sGradStream)
{
	if(m_hl < 0)
		return 0;
	return Matrix::MatsToStream(sGradStream, m_gs, m_hl+1); 
}


void SlaveTrainer::ReleaseWeight()
{
	if(m_ws)
	{
		delete [] m_ws; 
		m_ws = NULL; 
	} 
	if(m_gs)
	{
		delete [] m_gs; 
		m_gs = NULL; 
	}
	if(m_aos)
	{
		for(int32_t h = 0; h <= m_hl; h++) 
			delete m_aos[h]; 	
		delete [] m_aos; 
		m_aos = NULL; 
	}
	if(m_dos)
	{
		for(int32_t h = 0; h <= m_hl; h++) 
			delete m_dos[h]; 	
		delete [] m_dos; 
		m_dos = NULL; 
	}
	m_hl = -1; 
}


int32_t SlaveTrainer::GetHiddenLevels()
{
	return m_hl; 
} 


bool SlaveTrainer::SetCancelFlag(const bool bCancel)
{
	m_bCancelFlag = bCancel; 
	return m_bCancelFlag;  
}


bool SlaveTrainer::GetCancelFlag()
{
	return m_bCancelFlag;  
}


bool SlaveTrainer::CalcBatchGrad(double& dAvgLoss, const int32_t nBatchSize, const EActType eHiddenAct) 
{
	if(m_nPattOff == 0)
		PattShuffle(); 

	int32_t y_len = m_vtrPatts[m_nPattOff]->m_nYCnt;
	dAvgLoss = 0.0; 
	double loss;
	int32_t batch_size = (nBatchSize > 0 && nBatchSize < (int32_t)m_vtrPatts.size()) ? nBatchSize : (int32_t)m_vtrPatts.size();   
	
	for(int32_t t = 0; t < batch_size; t++) 
	{
		if(m_bCancelFlag)
			break; 
		if(!FeedForward(m_vtrPatts[m_nPattOff]->m_x, m_vtrPatts[m_nPattOff]->m_nXCnt, eHiddenAct))
			return false; 
		if(!BackPropagate(loss, m_vtrPatts[m_nPattOff]->m_x, m_vtrPatts[m_nPattOff]->m_nXCnt, m_vtrPatts[m_nPattOff]->m_y, y_len, eHiddenAct))
			return false;  
		dAvgLoss += loss; 
		m_nPattOff++; 
		if(m_nPattOff >= (int32_t)m_vtrPatts.size()) 
		{
			PattShuffle(); 
			m_nPattOff = 0; 
		} 
	}
	dAvgLoss /= (double)batch_size; 
	return true; 
}


bool SlaveTrainer::FeedForward(const double* x, const int32_t x_len, const EActType actHidden)
{
	if(!m_ws || !m_gs || !m_aos || !m_dos || !x)
		return false; 
	if(x_len != m_ws[0].Rows() - 1)
		return false;

	EActType e_out_act; 
	if(m_ws[m_hl].Cols() == 1)
		e_out_act = _ACT_LINEAR; 
	else if(m_ws[m_hl].Cols() == 2)
		e_out_act = _ACT_SIGMOID; 
	else
		e_out_act = _ACT_SOFTMAX; 

	if(m_hl == 0)
	{
		LayerActivation(m_aos[0], x, m_ws[0], e_out_act); 
	}
	else
	{ 
		for(int32_t h = 0; h <= m_hl; h++) 
		{
			if(h == 0)
				LayerActivation(m_aos[h], x, m_ws[h], actHidden); 
			else if(h == m_hl)
				LayerActivation(m_aos[h], m_aos[h-1], m_ws[h], e_out_act); 
			else
				LayerActivation(m_aos[h], m_aos[h-1], m_ws[h], actHidden); 
		}
	}
	
	return true; 
}


bool SlaveTrainer::BackPropagate(double& dLoss, const double* x, const int32_t x_len, const double* y, const int32_t y_len, 
		const EActType actHidden)
{
	if(!m_ws || !m_gs || !m_aos || !m_dos || !x || !y)
		return false; 
	if(x_len != m_ws[0].Rows() - 1 || y_len != m_ws[m_hl].Cols())
		return false; 

	// caculate delta of output layer
	for(int32_t j = 0; j < y_len; j++)
		m_dos[m_hl][j] = m_aos[m_hl][j] - y[j]; 

	if(m_hl == 0)
	{
		if(y_len == 1)
		{ // regression, Linear as output activation, quadratic as loss
			m_gs[0][x_len][0] += m_dos[0][0] * Activation::DActivate(m_aos[0][0], _ACT_LINEAR) * 1.0; 
			for(int32_t i = 0; i < x_len; i++)
				m_gs[0][i][0] += m_dos[0][0] * Activation::DActivate(m_aos[0][0], _ACT_LINEAR) * x[i]; 
			dLoss = Activation::Loss_Quadratic(m_aos[0][0], y[0]); // quadratic
		}
		else if(y_len == 2)
		{ // bi-classification, Sigmoid as output activation, cross entropy as loss
			m_gs[0][x_len][0] += m_dos[0][0] * 1.0; 
			for(int32_t i = 0; i < x_len; i++)
				m_gs[0][i][0] += m_dos[0][0] * x[i]; 
			dLoss = Activation::Loss_CrossEntropy(m_aos[0][0], y[0]); // cross entropy
		}	
		else
		{ // multi-classification, Softmax as output activation, log likelihood as loss
			for(int32_t j = 0; j < y_len; j++) 	
			{
				m_gs[0][x_len][j] += (m_aos[0][j] - 1.0) * 1.0; 
				for(int32_t i = 0; i < x_len; i++)
					m_gs[0][i][j] += (m_aos[i][j] - 1.0) * x[i]; 
			}
			dLoss = Activation::Loss_LogLikelihood(m_aos[0], y, y_len);  // log likelihood 
		}		
	}
	else
	{
		// delta back propagation
		if(y_len == 1 || y_len == 2) 
			LayerDeltaBack(m_dos[m_hl-1], m_dos[m_hl], m_ws[m_hl], true); 
		else
			LayerDeltaBack(m_dos[m_hl-1], m_dos[m_hl], m_ws[m_hl]); 
		for(int32_t h = m_hl - 2 ; h >= 0; h--)
			LayerDeltaBack(m_dos[h], m_dos[h+1], m_ws[h+1]); 
	
		// update the gradient matrix of loss function of output layer
		int32_t rows = m_gs[m_hl].Rows(); 
		int32_t cols = m_gs[m_hl].Cols(); 
		if(y_len == 1)
		{ // regression, Linear as output activation, quadratic as loss
			m_gs[m_hl][rows-1][0] += m_dos[m_hl][0] * Activation::DActivate(m_aos[m_hl][0], _ACT_LINEAR) * 1.0; 
			for(int32_t i = 0; i < rows - 1; i++) 
				m_gs[m_hl][i][0] += m_dos[m_hl][0] * Activation::DActivate(m_aos[m_hl][0], _ACT_LINEAR) * m_aos[m_hl-1][i]; 
			dLoss = Activation::Loss_Quadratic(m_aos[m_hl][0], y[0]); // quadratic
		}
		else if(y_len == 2)
		{ // bi-classification, Sigmoid as output activation, cross entropy as loss
			m_gs[m_hl][rows-1][0] += m_dos[m_hl][0] * 1.0; 
			for(int32_t i = 0; i < rows - 1; i++) 
				m_gs[m_hl][i][0] += m_dos[m_hl][0] * m_aos[m_hl-1][i]; 
			dLoss = Activation::Loss_CrossEntropy(m_aos[m_hl][0], y[0]); // cross entropy
		}	
		else
		{ // multi-classification, Softmax as output activation, log likelihood as loss
			for(int32_t j = 0; j < cols; j++) 
			{
				m_gs[m_hl][rows-1][j] += (m_aos[m_hl][j] - 1.0) * 1.0; 
				for(int32_t i = 0; i < rows - 1; i++) 
					m_gs[m_hl][i][j] += (m_aos[m_hl][j] - 1.0) * m_aos[m_hl-1][i]; 
			}
			dLoss = Activation::Loss_LogLikelihood(m_aos[m_hl], y, y_len);  // log likelihood 
		}		

		// update the gradient matrices of loss function of hidden layers
		for(int32_t h = m_hl - 1; h > 0; h--)
		{
			rows = m_gs[h].Rows(); 
			cols = m_gs[h].Cols();
			for(int32_t j = 0; j < cols; j++)
			{
				m_gs[h][rows-1][j] += m_dos[h][j] * Activation::DActivate(m_aos[h][j], actHidden) * 1.0; 
				for(int32_t i = 0; i < rows - 1; i++)
					m_gs[h][i][j] += m_dos[h][j] * Activation::DActivate(m_aos[h][j], actHidden) * m_aos[h-1][i]; 
			}
		}
		// the lowest hidden layer
		rows = m_gs[0].Rows(); 
		cols = m_gs[0].Cols();
		for(int32_t j = 0; j < cols; j++)
		{
			m_gs[0][rows-1][j] += m_dos[0][j] * Activation::DActivate(m_aos[0][j], actHidden) * 1.0; 
			for(int32_t i = 0; i < rows - 1; i++)
				m_gs[0][i][j] += m_dos[0][j] * Activation::DActivate(m_aos[0][j], actHidden) * x[i]; 
		}
	}

	return true; 
}


bool SlaveTrainer::LayerActivation(double* up_ao, const double* low_ao, Matrix& w, const EActType eActType)
{
	if(!up_ao || !low_ao || w.IsNull())
		return false; 

	double e = 0.0; 
	for(int32_t j = 0; j < w.Cols(); j++) 
	{
		// forward propagation
		up_ao[j] = w[w.Rows()-1][j]; 	// bias
		for(int32_t i = 0; i < w.Rows() - 1; i++)
			up_ao[j] += w[i][j] * low_ao[i];

		// activation	
		if(eActType == _ACT_SOFTMAX)
			e += exp(up_ao[j]); 
		else
			up_ao[j] = Activation::Activate(up_ao[j], eActType); 
	}	
	if(eActType == _ACT_SOFTMAX)
	{ // softmax
		for(int32_t j = 0; j < w.Cols(); j++) 
			up_ao[j] = exp(up_ao[j]) / e; 
	}

	return true; 
}


bool SlaveTrainer::LayerDeltaBack(double* low_do, const double* up_do, Matrix& w, const bool bOneCol)
{
	if(!low_do || !up_do || w.IsNull())
		return false; 

	for(int32_t i = 0; i < w.Rows() - 1; i++) 
	{ 
		low_do[i] = up_do[0] * w[i][0];
		if(!bOneCol) 
		{
			for(int32_t j = 1; j < w.Cols(); j++) 
				low_do[i] += up_do[j] * w[i][j]; 
		}
	}

	return true; 
}




