#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

 
// Gray Modulation
extern float   Map4QAM[2];// = { -0.707107f, 0.707107f };//sqrt(0.5)
extern float	Map16QAM[4];// = {-0.316228f, -0.948683f,  0.316228f,  0.948683f};//sqrt(0.1) sqrt(0.9)
extern float	Map64QAM[8];// = {-0.462910f, -0.154303f, -0.771517f, -1.08012f, 0.462910f,  0.154303f,  0.771517f,  1.08012f};//

void QAM_Modulation(int	*inbit,
	        float	*Sig_I, 
		    float	*Sig_Q,   
		    int		SymbolBitN, 
		    int		SigLen)
{          
	int	M;
	int	i,j;
	int	Tmp_I, Tmp_Q;
	float *MapTable=NULL;
	
	M = SymbolBitN/2;
	
	switch (SymbolBitN)
	{
		case 2:
			MapTable = Map4QAM;
			break;
		case 4:
			MapTable = Map16QAM;
			break;
		case 6:
			MapTable = Map64QAM;
			break;
		default: printf("Invalid SymbolitN");
	}
	
	for(i=0; i<SigLen; i++)
	{
		Tmp_I = 0;
		Tmp_Q = 0;
		
		for(j=0; j<M; j++)
		{
			Tmp_I = Tmp_I + (inbit[i*SymbolBitN+j]<<(M-j-1));
			Tmp_Q = Tmp_Q + (inbit[i*SymbolBitN+M+j]<<(M-j-1));
		}
		
		Sig_I[i] = MapTable[Tmp_I];
		Sig_Q[i] = MapTable[Tmp_Q];
		
	}	
}

void QAM_Demodulation(float	*LLRD, 
		      float	*Recv_I, 
		      float	*Recv_Q, 
		      float	Sigma2, 
		      int	SigLen, 
		      int	SymbolBitN, 
		      int	TxAntNum)
{
	int	n_Tx, n_Sig;
	
	float vari;
	
	switch (SymbolBitN)
	{
		case 2:
			for(n_Sig=0; n_Sig<SigLen; n_Sig++)
			{
				for(n_Tx=0; n_Tx<TxAntNum; n_Tx++)
				{
					vari = 1;// 2.828427 / Sigma2;  //4/sqrt(2)     
					
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+0] = Recv_I[n_Sig*TxAntNum+n_Tx]*vari;  
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+1] = Recv_Q[n_Sig*TxAntNum+n_Tx]*vari;
					
				}
			}
			break;
		case 4:
			for(n_Sig=0; n_Sig<SigLen; n_Sig++)
			{
				for(n_Tx=0; n_Tx<TxAntNum; n_Tx++)
				{
					vari = 1;// 1.264911 / Sigma2;		// 4/sqrt(10)
					
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+0] = Recv_I[n_Sig*TxAntNum+n_Tx]*vari;
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+1] = fabs(LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+0])-vari*0.6324555;	// 2/sqrt(10)
					
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+2] = Recv_Q[n_Sig*TxAntNum+n_Tx]*vari;
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+3] = fabs(LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+2])-vari*0.6324555;
				}
					
			}	
			break;
		case 6:
			
			for(n_Sig=0; n_Sig<SigLen; n_Sig++)
			{
				for(n_Tx=0; n_Tx<TxAntNum; n_Tx++)
				{
					vari = 1;//0.6172134 / Sigma2;		// 4/sqrt(42)
					
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+0] = Recv_I[n_Sig*TxAntNum+n_Tx]*vari;
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+1] = fabs(LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+0]) - vari*0.6172134;
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+2] = fabs(LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+1]) - vari*0.3086067;	// 2/sqrt(42)
					
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+3] = Recv_Q[n_Sig*TxAntNum+n_Tx]*vari;
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+4] = fabs(LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+3]) - vari*0.6172134;
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+5] = fabs(LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+4]) - vari*0.3086067;
				}
			
			}
			break;
		default: printf("Invalid SymbolBitN!\n");
	}
}





