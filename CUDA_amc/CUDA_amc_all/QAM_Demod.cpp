#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <random>
 
//输出的是似然比
float   Map4QAM[2] = {-0.707107f, 0.707107f};
float	Map16QAM[4] = {-0.316228f, -0.948683f,  0.316228f,  0.948683f};
float	Map64QAM[8] = {-0.462910f, -0.154303f, -0.771517f, -1.080123f,
		        0.462910f,  0.154303f,  0.771517f,  1.080123f};
float	Map256QAM[16] = {-0.383482f, -0.536875f, -0.230089f, -0.076696f,
			 -0.843661f, -0.690268f, -0.997054f, -1.150447f,
			  0.383482f,  0.536875f,  0.230089f,  0.076696f,
			  0.843661f,  0.690268f,  0.997054f,  1.150447f};


float	maxo(float a, float b)
{
	//return (a>b)? a:b;
	return (a>b)? a+log(1+exp(-fabs(a-b))):b+log(1+exp(-fabs(a-b)));
}
float x=2;

void QAM_Demodulation(float	*LLRD, 
		      float	*Recv_I, 
		      float	*Recv_Q, 
		      float	*Sigma2, 
		      float	*LLRA, 
		      int	SigLen, 
		      int	SymbolBitN, 
		      int	TxAntNum)
{
	int	n_Tx, n_Sig, n_bit, n_const;
	int	M, ConstNum;
	
	float	nom[4], denom[4], inf;   //这里的软解调是利用之前四个比特的LLR
	float	oldLLR[4];
	float	metric0, metric1, metric;  //三种度量
	
	float	Xi, Yi, vari;
	
	M = SymbolBitN/2;   //因为分成两路，所以要除以2
	ConstNum = (1<<M);  //相当于2的M次方  const代表map里面的下标
	inf = 1e6;   
	
	switch (SymbolBitN)
	{
		case 2:      //QAM调制
			for(n_Sig=0; n_Sig<SigLen; n_Sig++)
			{
				for(n_Tx=0; n_Tx<TxAntNum; n_Tx++)
				{
					Xi = Recv_I[n_Sig*TxAntNum+n_Tx]; // 将接收到符号的I， Q分别映射映射到每一个天线上 Xi指实部 Yi指虚部
					Yi = Recv_Q[n_Sig*TxAntNum+n_Tx];
					vari = Sigma2[n_Sig*TxAntNum+n_Tx];  //sigma2 应该是指噪声方差
					
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+0] = 4*Xi/sqrt(x)/vari;  //因为QAM的两个值是互为相反数的，所以此为最大似然比
					LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+1] = 4*Yi/sqrt(x)/vari;
					
				}
			}
			break;
		case 4:
			for(n_Sig=0; n_Sig<SigLen; n_Sig++)
			{
				for(n_Tx=0; n_Tx<TxAntNum; n_Tx++)
				{
					Xi = Recv_I[n_Sig*TxAntNum+n_Tx];
					Yi = Recv_Q[n_Sig*TxAntNum+n_Tx];
					vari = Sigma2[n_Sig*TxAntNum+n_Tx];
					
					// In-pase Component  同相分量
					for(n_bit=0; n_bit<M; n_bit++)
					{
						nom[n_bit] = -inf;
						denom[n_bit] = -inf;
						oldLLR[n_bit] = LLRA[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit];   //这里的LLRA存储的应该是已经求出信息的似然比，是由SISO模块求出来的对应于比特的似然比，然后再经过交织。
					}
					
					for(n_const=0; n_const<ConstNum; n_const++)
					{
						metric0 = 0;
						metric1 = 0;
						
						metric0 = -(Xi-Map16QAM[n_const])*(Xi-Map16QAM[n_const])/vari;  //整个符号的检测
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							metric1 = metric1 + (2*((n_const>>(M-1-n_bit))&1)-1)*oldLLR[n_bit];   //  相当于（4-7）里的后面去掉第i为的部分
						}
						
						metric =float( metric0 + 0.5*metric1);     //这个相当于（4-7）里的分子分母部分
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							if ((n_const>>(M-1-n_bit))&1)
							{
								nom[n_bit] = maxo(nom[n_bit], metric);   //当第n个bit为1时的似然函数
							}
							else
							{
								denom[n_bit] = maxo(denom[n_bit], metric);   //当第n个bit为0时的似然函数
							}
						}
					}
					
					for(n_bit=0; n_bit<M; n_bit++)
					{
						LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit] = nom[n_bit] - denom[n_bit] - oldLLR[n_bit];  //采用的是软干扰抵消迭代算法
					}
					
					// Quadure Component  正交分量
					for(n_bit=0; n_bit<M; n_bit++)
					{
						nom[n_bit] = -inf;
						denom[n_bit] = -inf;
						oldLLR[n_bit] = LLRA[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit+M];
					}
					
					for(n_const=0; n_const<ConstNum; n_const++)
					{
						metric0 = 0;
						metric1 = 0;
						
						metric0 = -(Yi-Map16QAM[n_const])*(Yi-Map16QAM[n_const])/vari;
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							metric1 = metric1 + (2*((n_const>>(M-1-n_bit))&1)-1)*oldLLR[n_bit];
						}
						
						metric =float( metric0 + 0.5*metric1);
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							if ((n_const>>(M-1-n_bit))&1)
							{
								nom[n_bit] = maxo(nom[n_bit], metric);
							}
							else
							{
								denom[n_bit] = maxo(denom[n_bit], metric);
							}
						}
					}
					
					for(n_bit=0; n_bit<M; n_bit++)
					{
						LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit+M] = nom[n_bit] - denom[n_bit] - oldLLR[n_bit];
					}
					
				}
			}
			break;
		case 6:
			for(n_Sig=0; n_Sig<SigLen; n_Sig++)
			{
				for(n_Tx=0; n_Tx<TxAntNum; n_Tx++)
				{
					Xi = Recv_I[n_Sig*TxAntNum+n_Tx];
					Yi = Recv_Q[n_Sig*TxAntNum+n_Tx];
					vari = Sigma2[n_Sig*TxAntNum+n_Tx];
					
					// In-pase Component
					for(n_bit=0; n_bit<M; n_bit++)
					{
						nom[n_bit] = -inf;
						denom[n_bit] = -inf;
						oldLLR[n_bit] = LLRA[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit];
					}
					
					for(n_const=0; n_const<ConstNum; n_const++)
					{
						metric0 = 0;
						metric1 = 0;
						
						metric0 = -(Xi-Map64QAM[n_const])*(Xi-Map64QAM[n_const])/vari;
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							metric1 = metric1 + (2*((n_const>>(M-1-n_bit))&1)-1)*oldLLR[n_bit];
						}
						
						metric =float( metric0 + 0.5*metric1);
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							if ((n_const>>(M-1-n_bit))&1)
							{
								nom[n_bit] = maxo(nom[n_bit], metric);
							}
							else
							{
								denom[n_bit] = maxo(denom[n_bit], metric);
							}
						}
					}
					
					for(n_bit=0; n_bit<M; n_bit++)
					{
						LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit] = nom[n_bit] - denom[n_bit] - oldLLR[n_bit];
					}
					
					// Quadure Component
					for(n_bit=0; n_bit<M; n_bit++)
					{
						nom[n_bit] = -inf;
						denom[n_bit] = -inf;
						oldLLR[n_bit] = LLRA[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit+M];
					}
					
					for(n_const=0; n_const<ConstNum; n_const++)
					{
						metric0 = 0;
						metric1 = 0;
						
						metric0 = -(Yi-Map64QAM[n_const])*(Yi-Map64QAM[n_const])/vari;
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							metric1 = metric1 + (2*((n_const>>(M-1-n_bit))&1)-1)*oldLLR[n_bit];
						}
						
						metric =float( metric0 + 0.5*metric1);
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							if ((n_const>>(M-1-n_bit))&1)
							{
								nom[n_bit] = maxo(nom[n_bit], metric);
							}
							else
							{
								denom[n_bit] = maxo(denom[n_bit], metric);
							}
						}
					}
					
					for(n_bit=0; n_bit<M; n_bit++)
					{
						LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit+M] = nom[n_bit] - denom[n_bit] - oldLLR[n_bit];
					}
					
				}
			}
			break;
		case 8:
			for(n_Sig=0; n_Sig<SigLen; n_Sig++)
			{
				for(n_Tx=0; n_Tx<TxAntNum; n_Tx++)
				{
					Xi = Recv_I[n_Sig*TxAntNum+n_Tx];
					Yi = Recv_Q[n_Sig*TxAntNum+n_Tx];
					vari = Sigma2[n_Sig*TxAntNum+n_Tx];
					
					// In-pase Component
					for(n_bit=0; n_bit<M; n_bit++)
					{
						nom[n_bit] = -inf;
						denom[n_bit] = -inf;
						oldLLR[n_bit] = LLRA[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit];
					}
					
					for(n_const=0; n_const<ConstNum; n_const++)
					{
						metric0 = 0;
						metric1 = 0;
						
						metric0 = -(Xi-Map256QAM[n_const])*(Xi-Map256QAM[n_const])/vari;
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							metric1 = metric1 + (2*((n_const>>(M-1-n_bit))&1)-1)*oldLLR[n_bit];
						}
						
						metric = float(metric0 + 0.5*metric1);
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							if ((n_const>>(M-1-n_bit))&1)
							{
								nom[n_bit] = maxo(nom[n_bit], metric);
							}
							else
							{
								denom[n_bit] = maxo(denom[n_bit], metric);
							}
						}
					}
					
					for(n_bit=0; n_bit<M; n_bit++)
					{
						LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit] = nom[n_bit] - denom[n_bit] - oldLLR[n_bit];
					}
					
					// Quadure Component
					for(n_bit=0; n_bit<M; n_bit++)
					{
						nom[n_bit] = -inf;
						denom[n_bit] = -inf;
						oldLLR[n_bit] = LLRA[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit+M];
					}
					
					for(n_const=0; n_const<ConstNum; n_const++)
					{
						metric0 = 0;
						metric1 = 0;
						
						metric0 = -(Yi-Map256QAM[n_const])*(Yi-Map256QAM[n_const])/vari;
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							metric1 = metric1 + (2*((n_const>>(M-1-n_bit))&1)-1)*oldLLR[n_bit];
						}
						
						metric = float(metric0 + 0.5*metric1);
						
						for(n_bit=0; n_bit<M; n_bit++)
						{
							if ((n_const>>(M-1-n_bit))&1)
							{
								nom[n_bit] = maxo(nom[n_bit], metric);
							}
							else
							{
								denom[n_bit] = maxo(denom[n_bit], metric);
							}
						}
					}
					
					for(n_bit=0; n_bit<M; n_bit++)
					{
						LLRD[n_Sig*TxAntNum*SymbolBitN+n_Tx*SymbolBitN+n_bit+M] = nom[n_bit] - denom[n_bit] - oldLLR[n_bit];
					}
					
				}
			}
			break;
		default: printf("Invalid SymbolBitN!\n");
	}
}

