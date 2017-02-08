#ifndef _METIS_PLATFORM_SLAVE_TRAINER_H 
#define _METIS_PLATFORM_SLAVE_TRAINER_H 


#include <string>
#include <vector>
using namespace std; 
#include "Pattern.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
using namespace metis_nn;  


namespace metis_plat
{

typedef struct _patts_info_t
{
	int32_t x_dims; 
	int32_t y_dims; 
	int32_t patts; 
} PattsInfoT;


class SlaveTrainer
{
public:
	SlaveTrainer(); 
	virtual ~SlaveTrainer(); 

	bool PushPatt_inString(const char* sPattStr); 
	bool PushPatt_inStream(const char* sPattStream, const int32_t nLen); 
	int32_t PattCnt(); 
	PattsInfoT GetPattsInfo(); 
	void PattShuffle(); 
	void ReleasePatts(); 

	bool SetWeight_inStream(const char* sWeightStream, const int32_t nLen); 
	int GetGrad_asStream(char* sGradStream);
	int32_t WeightStreamLen(); 
	void ReleaseWeight(); 
	int32_t GetHiddenLevels();  

	bool SetCancelFlag(const bool bCancel); 
	bool GetCancelFlag(); 	

	bool CalcBatchGrad(double& dAvgLoss, const int32_t nBatchSize, const EActType eHiddenAct = _ACT_RELU);  

private: 
	bool FeedForward(const double* x, const int32_t x_len, const EActType actHidden = _ACT_RELU);
	bool BackPropagate(double& dLoss, const double* x, const int32_t x_len, const double* y, const int32_t y_len, 
		const EActType actHidden = _ACT_RELU);

	bool LayerActivation(double* up_ao, const double* low_ao, Matrix& w, const EActType eActType); 
	bool LayerDeltaBack(double* low_do, const double* up_do, Matrix& w, const bool bOneCol = false); 
	//bool GradientInLayer(const double* low_ao, const double* up_ao, const double* up_do, Matrix& g, const EActType eActType, bool bOneCol = false);

private: 
	Matrix* m_ws; 
	Matrix* m_gs;	// gradient matrix of loss function 
	double** m_aos;
	double** m_dos;
	int32_t m_hl; 
	
	vector<Pattern*> m_vtrPatts; 
	int32_t m_nPattOff; 
	static bool m_bCancelFlag; 
};

}


#endif /* _METIS_PLATFORM_SLAVE_TRAINER_H */

