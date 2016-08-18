#include <stdio.h>
#include <malloc.h>
#include "Turbo.h"
/*library of GPU*/
#include "math_functions.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "device_functions.h"
#include "cuda_texture_types.h"

/*pre-set parameters chosed with CQI*/
#define win_N  24						//inner Parallel Sliding Window

#define win_L  100
#define half_win_L 50
#define info_L  2400
#define info_tL 2432
#define block_N 44
#define ab_thread_N 384
#define half_abt_N 192
#define info_thread_N 400
#define info_thread_tN 403
#define cal_time 6
#define bank_size 32

/*
输出参数：
dev_gamma：		gamma；

输入参数：
dev_a：			先验信息；
dev_s：			系统信息；
dev_p：			校验信息；
*/
__global__ void extMapKernel(float *dev_llr, float *dev_a, float *dev_s, int *dev_inter, int interleave_type)
{
	__shared__ float shr_a[info_L];

	unsigned int a_idx, s_idx, i_idx, i;
	i_idx = threadIdx.x;
	a_idx = blockIdx.x*info_L;
	s_idx = blockIdx.x*info_tL;

	if (interleave_type == 0)
	{
		for (i = 0; i < cal_time; i++)
		{
			shr_a[i_idx] = dev_llr[a_idx + i_idx] - dev_s[s_idx + i_idx] - dev_a[a_idx + i_idx];
			i_idx += info_thread_N;
		}
		__syncthreads();
		for (i = 0; i < cal_time; i++)
		{
			i_idx -= info_thread_N;
			dev_a[a_idx + i_idx] = shr_a[dev_inter[i_idx]];
		}
	}
	else
	{
		for (i = 0; i < cal_time; i++)
		{
			shr_a[dev_inter[i_idx]] = dev_llr[a_idx + i_idx] - dev_s[s_idx + i_idx] - dev_a[a_idx + i_idx];
			i_idx += info_thread_N;
		}
		__syncthreads();
		for (i = 0; i < cal_time; i++)
		{
			i_idx -= info_thread_N;
			dev_a[a_idx + i_idx] = shr_a[i_idx];
		}
	}
}
__global__ void abMapKernel(float *dev_llr, float *dev_gamma, float *dev_ab, float *last_alfa, float *last_beta, int *dev_para, int iteration) //last iteration value
{
	__shared__ float shr_8illr[half_abt_N << 3];
	__shared__ float shr_8jllr[half_abt_N << 3];
	float plus, minus, gamma, ab;

	unsigned int half_idx = threadIdx.x%half_abt_N;
	unsigned int win_n = half_idx >> 3;
	unsigned int state_n = half_idx & 7;
	unsigned int i;
	int add_loc, sub_loc;
	float *g_ptr, *ab_ptr;

	if (threadIdx.x < half_abt_N)
	{
		/*get parameter from mem*/
		ab_ptr = dev_ab + ((blockIdx.x*info_L << 3) + half_idx);
		g_ptr = dev_gamma + (((blockIdx.x << 1) + dev_para[state_n])*info_tL + win_n*win_L);
		add_loc = dev_para[16 + state_n];
		sub_loc = dev_para[32 + state_n];
		/*get last alfa*/
		if (iteration == 0)
		{
			ab = state_n == 0 ? 0.0f : -10000.0f;

			if (win_n > 0)
			{
				for (i = 3; i >0; i--)
				{
					gamma = *(g_ptr - i);
					plus = __shfl(ab, add_loc, 8) + gamma;
					minus = __shfl(ab, sub_loc, 8) - gamma;
					ab = logf(expf(plus) + expf(minus));
				}
			}
		}
		else
		{
			ab = last_alfa[blockIdx.x*half_abt_N + half_idx];
		}

		/*calculate half alfa*/
		for (i = 0; i < half_win_L; i++)
		{
			*ab_ptr = ab;
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = logf(expf(plus) + expf(minus));
			ab_ptr += half_abt_N;
			g_ptr++;
		}
	}
	else
	{
		/*get parameter from mem*/
		ab_ptr = dev_ab + ((blockIdx.x*info_L << 3) + (win_L - 1)*half_abt_N + half_idx);
		g_ptr = dev_gamma + (((blockIdx.x << 1) + dev_para[8 + state_n])*info_tL + win_n*win_L + win_L - 1);
		add_loc = dev_para[24 + state_n];
		sub_loc = dev_para[40 + state_n];

		/*get last beta*/
		if (iteration == 0)
		{
			int v_L = win_n == win_N - 1 ? 3 : 20;
			ab = state_n == 0 ? 0.0f : -10000.0f;
			for (i = v_L; i >0; i--)
			{
				gamma = *(g_ptr + i);
				plus = __shfl(ab, add_loc, 8) + gamma;
				minus = __shfl(ab, sub_loc, 8) - gamma;
				ab = logf(expf(plus) + expf(minus));
			}
			if (win_n == win_N - 1)
			{
				last_beta[blockIdx.x*half_abt_N + half_idx] = ab;
			}
		}
		else
		{
			ab = last_beta[blockIdx.x*half_abt_N + half_idx];
		}

		/*calculate half beta*/
		for (i = 0; i < half_win_L; i++)
		{
			*ab_ptr = ab;
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = logf(expf(plus) + expf(minus));

			ab_ptr -= half_abt_N;
			g_ptr--;
		}
	}
	__syncthreads();
	/*calculate rest alfa\beta and prepare for llr*/

	if (threadIdx.x < half_abt_N)
	{
		float minus_tmp, plus_tmp;
		float *a_ptr = dev_llr + (blockIdx.x*info_L + win_n* win_L + half_win_L + state_n);
		float *shr_ptr = shr_8illr + ((win_n << 3) + state_n*half_abt_N);
		unsigned int tmp_loc = state_n & 3;

		for (int count = 0; i < win_L; i++, count++)
		{
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = logf(expf(plus) + expf(minus));

			minus = *ab_ptr + minus;
			plus = *ab_ptr + plus;

			minus_tmp = __shfl_down(minus, 4, 8);
			plus_tmp = __shfl_up(plus, 4, 8);
			if (state_n <4)
			{
				shr_8illr[count*half_abt_N + half_idx] = logf(expf(minus_tmp) + expf(minus));
			}
			else
			{
				shr_8illr[count*half_abt_N + half_idx] = logf(expf(plus_tmp) + expf(plus));
			}

			if (count == 7 || i == win_L - 1 && state_n<(half_win_L & 7))
			{
				count = -1;

				*a_ptr = logf(expf(*(shr_ptr + 4 + tmp_loc)) + expf(*(shr_ptr + 4 + (tmp_loc + 1 & 3))) + expf(*(shr_ptr + 4 + (tmp_loc + 2 & 3))) + expf(*(shr_ptr + 4 + (tmp_loc + 3 & 3)))) - logf(expf(*(shr_ptr + tmp_loc)) + expf(*(shr_ptr + (tmp_loc + 1 & 3))) + expf(*(shr_ptr + (tmp_loc + 2 & 3))) + expf(*(shr_ptr + (tmp_loc + 3 & 3))));
				a_ptr += 8;
			}
			ab_ptr += half_abt_N;
			g_ptr++;
		}
		if (win_n < win_N - 1)
		{
			last_alfa[blockIdx.x*half_abt_N + half_idx + 8] = ab;
		}
	}
	else
	{
		float minus_tmp, plus_tmp;
		float *a_ptr = dev_llr + (blockIdx.x*info_L + win_n* win_L + half_win_L - 1 - state_n);
		float *shr_ptr = shr_8jllr + ((win_n << 3) + state_n*half_abt_N);
		unsigned int tmp_loc = state_n & 3;

		for (int count = 0; i < win_L; i++, count++)
		{
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = logf(expf(plus) + expf(minus));

			minus = *ab_ptr + minus;
			plus = *ab_ptr + plus;

			minus_tmp = __shfl_down(minus, 4, 8);
			plus_tmp = __shfl_up(plus, 4, 8);
			if (state_n <4)
			{
				shr_8jllr[count*half_abt_N + half_idx] = logf(expf(minus_tmp) + expf(minus));
			}
			else
			{
				shr_8jllr[count*half_abt_N + half_idx] = logf(expf(plus_tmp) + expf(plus));
			}

			if (count == 7 || i == win_L - 1 && state_n<(half_win_L & 7))
			{
				count = -1;
				*a_ptr = logf(expf(*(shr_ptr + 4 + tmp_loc)) + expf(*(shr_ptr + 4 + (tmp_loc + 1 & 3))) + expf(*(shr_ptr + 4 + (tmp_loc + 2 & 3))) + expf(*(shr_ptr + 4 + (tmp_loc + 3 & 3)))) - logf(expf(*(shr_ptr + tmp_loc)) + expf(*(shr_ptr + (tmp_loc + 1 & 3))) + expf(*(shr_ptr + (tmp_loc + 2 & 3))) + expf(*(shr_ptr + (tmp_loc + 3 & 3))));
				a_ptr -= 8;
			}
			ab_ptr -= half_abt_N;
			g_ptr--;
		}
		if (win_n > 0)
		{
			last_beta[blockIdx.x*half_abt_N + half_idx - 8] = ab;
		}
	}
}

__global__ void gammaKernel(float *dev_gamma, float *dev_s, float *dev_p, float *dev_a)
{
	float gamma, p;
	unsigned g_idx, a_idx, s_idx, i;

	if (threadIdx.x < info_thread_N)
	{
		s_idx = blockIdx.x*info_tL + threadIdx.x;
		g_idx = (blockIdx.x*info_tL << 1) + threadIdx.x;
		a_idx = blockIdx.x*info_L + threadIdx.x;

		if (dev_a != NULL)
		{
			for (i = 0; i < cal_time; i++)
			{
				p = dev_p[s_idx];
				gamma = (dev_s[s_idx] - p + dev_a[a_idx])*0.5f;
				dev_gamma[g_idx] = gamma;
				dev_gamma[g_idx + info_tL] = gamma + p;
				s_idx += info_thread_N;
				a_idx += info_thread_N;
				g_idx += info_thread_N;
			}
		}
		else
		{
			for (i = 0; i < cal_time; i++)
			{
				p = dev_p[s_idx];
				gamma = (dev_s[s_idx] - p)*0.5f;
				dev_gamma[g_idx] = gamma;
				dev_gamma[g_idx + info_tL] = gamma + p;
				s_idx += info_thread_N;
				g_idx += info_thread_N;
			}
		}
	}
	else
	{
		s_idx = blockIdx.x*info_tL + info_L + threadIdx.x - info_thread_N;
		g_idx = (blockIdx.x*info_tL<<1) + info_L + threadIdx.x - info_thread_N;

		p = dev_p[s_idx];
		gamma = (dev_s[s_idx] - p)*0.5f;
		dev_gamma[g_idx] = gamma;
		dev_gamma[g_idx + info_tL] = gamma + p;
	}
}

__global__ void extKernel(float *dev_llr, float *dev_a, float *dev_s, int *dev_inter, int interleave_type)
{
	__shared__ float shr_a[info_L];

	unsigned int a_idx, s_idx, i_idx;
	i_idx = threadIdx.x;
	a_idx = blockIdx.x*info_L;
	s_idx = blockIdx.x*info_tL;

	if (interleave_type == 0)
	{
		for (i_idx = threadIdx.x; i_idx < info_L; i_idx += info_thread_N)
		{
			shr_a[i_idx] = 0.7f*(dev_llr[a_idx + i_idx] - dev_s[s_idx + i_idx] - dev_a[a_idx + i_idx]);
			
		}
		__syncthreads();
		for (i_idx = threadIdx.x; i_idx < info_L; i_idx += info_thread_N)
		{			
			dev_a[a_idx + i_idx] = shr_a[dev_inter[i_idx]];
		}
	}
	else
	{
		for (i_idx = threadIdx.x; i_idx < info_L; i_idx += info_thread_N)
		{
			shr_a[dev_inter[i_idx]] = 0.7f*(dev_llr[a_idx + i_idx] - dev_s[s_idx + i_idx] - dev_a[a_idx + i_idx]);
		}
		__syncthreads();
		for (i_idx = threadIdx.x; i_idx < info_L; i_idx += info_thread_N)
		{
			dev_a[a_idx + i_idx] = shr_a[i_idx];
		}
	}
}

__global__ void ab2Kernel(float *dev_llr, float *dev_gamma, float *dev_ab, float *last_alfa, float *last_beta, int *dev_para, int iteration) //last iteration value
{
	__shared__ float shr_8illr[half_abt_N << 3];
	__shared__ float shr_8jllr[half_abt_N << 3];

	float plus, minus, gamma, ab;

	unsigned int half_idx = threadIdx.x%half_abt_N;
	unsigned int win_n = half_idx >> 3;
	unsigned int state_n = half_idx & 7;
	unsigned int i;
	int add_loc, sub_loc;
	float *g_ptr, *ab_ptr;

	//parameter set and pretrain
	if (threadIdx.x < half_abt_N)
	{
		/*get parameter from mem*/
		ab_ptr = dev_ab + ((blockIdx.x*info_L << 3) + half_idx);
		g_ptr = dev_gamma + (((blockIdx.x << 1) + dev_para[state_n])*info_tL + win_n*win_L);
		add_loc = dev_para[16 + state_n];
		sub_loc = dev_para[32 + state_n];
		/*get last alfa*/
		if (iteration == 0)
		{
			ab = state_n == 0 ? 0.0f : -10000.0f;

			if (win_n > 0)
			{
				for (i = 20; i >0; i--)
				{
					gamma = *(g_ptr - i);
					plus = __shfl(ab, add_loc, 8) + gamma;
					minus = __shfl(ab, sub_loc, 8) - gamma;
					ab = fmaxf(plus, minus);
				}
			}
		}
		else
		{
			ab = last_alfa[(blockIdx.x*half_abt_N) + half_idx];
		}	
	}
	else
	{
		/*get parameter from mem*/
		ab_ptr = dev_ab + ((blockIdx.x*info_L << 3) + (win_L - 1)*half_abt_N + half_idx);
		g_ptr = dev_gamma + (((blockIdx.x << 1) + dev_para[8 + state_n])*info_tL + win_n*win_L + win_L - 1);
		add_loc = dev_para[24 + state_n];
		sub_loc = dev_para[40 + state_n];

		/*get last beta*/
		if (iteration == 0)
		{
			int v_L = win_n == win_N - 1 ? 3 : 20;
			ab = state_n == 0 ? 0.0f : -10000.0f;
			for (i = v_L; i >0; i--)
			{
				gamma = *(g_ptr + i);
				plus = __shfl(ab, add_loc, 8) + gamma;
				minus = __shfl(ab, sub_loc, 8) - gamma;
				ab = fmaxf(plus, minus);
			}
			if (win_n == win_N - 1)
			{
				last_beta[(blockIdx.x*half_abt_N) + half_idx] = ab;
			}
		}
		else
		{
			ab = last_beta[(blockIdx.x*half_abt_N) + half_idx];
		}		
	}

	//calculate half alfa beta
	if (threadIdx.x < half_abt_N)
	{
		/*calculate half alfa*/
		for (i = 0; i < half_win_L; i++)
		{
			*ab_ptr = ab;
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = fmaxf(plus, minus);
			ab_ptr += half_abt_N;
			g_ptr++;
			
		}
	}
	else
	{
		/*calculate half beta*/
		for (i = 0; i < half_win_L; i++)
		{
			*ab_ptr = ab;
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = fmaxf(plus, minus);
			ab_ptr -= half_abt_N;
			g_ptr--;
		}
	}
	__syncthreads();
	/*calculate rest alfa\beta and prepare for llr*/

	if (threadIdx.x < half_abt_N)
	{
		float minus_tmp, plus_tmp;
		float *a_ptr = dev_llr + (blockIdx.x*info_L + win_n* win_L + half_win_L + state_n);
		float *shr_ptr = shr_8illr + ((win_n << 3) + state_n*half_abt_N);
		unsigned int tmp_loc = state_n & 3;

		for (int count = 0; i < win_L; i++, count++)
		{
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = fmaxf(plus, minus);

			minus = *ab_ptr + minus;
			plus = *ab_ptr + plus;

			minus_tmp = __shfl_down(minus, 4, 8);
			plus_tmp = __shfl_up(plus, 4, 8);
			if (state_n <4)
			{
				shr_8illr[count*half_abt_N + half_idx] = fmaxf(minus_tmp, minus);
			}
			else
			{
				shr_8illr[count*half_abt_N + half_idx] = fmaxf(plus_tmp, plus);
			}

			if (count == 7 || i == win_L - 1 && state_n<(half_win_L & 7))
			{
				count = -1;

				*a_ptr = fmaxf(fmaxf(*(shr_ptr + 4 + tmp_loc), *(shr_ptr + 4 + (tmp_loc + 1 & 3))), fmaxf(*(shr_ptr + 4 + (tmp_loc + 2 & 3)), *(shr_ptr + 4 + (tmp_loc + 3 & 3)))) - fmaxf(fmaxf(*(shr_ptr + tmp_loc), *(shr_ptr + (tmp_loc + 1 & 3))), fmaxf(*(shr_ptr + (tmp_loc + 2 & 3)), *(shr_ptr + (tmp_loc + 3 & 3))));
				a_ptr += 8;
			}
			ab_ptr += half_abt_N;
			g_ptr++;
		}
		if (win_n < win_N - 1)
		{
			last_alfa[blockIdx.x*half_abt_N + half_idx + 8] = ab;
		}
	}
	else
	{
		float minus_tmp, plus_tmp;
		float *a_ptr = dev_llr + (blockIdx.x*info_L + win_n* win_L + half_win_L - 1 - state_n);
		float *shr_ptr = shr_8jllr + ((win_n << 3) + state_n*half_abt_N);
		unsigned int tmp_loc = state_n & 3;

		for (int count = 0; i < win_L; i++, count++)
		{
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = fmaxf(plus, minus);

			minus = *ab_ptr + minus;
			plus = *ab_ptr + plus;

			minus_tmp = __shfl_down(minus, 4, 8);
			plus_tmp = __shfl_up(plus, 4, 8);
			if (state_n <4)
			{
				shr_8jllr[count*half_abt_N + half_idx] = fmaxf(minus_tmp, minus);
			}
			else
			{
				shr_8jllr[count*half_abt_N + half_idx] = fmaxf(plus_tmp, plus);
			}

			if (count == 7 || i == win_L - 1 && state_n<(half_win_L & 7))
			{
				count = -1;

				*a_ptr = fmaxf(fmaxf(*(shr_ptr + 4 + tmp_loc), *(shr_ptr + 4 + (tmp_loc + 1 & 3))), fmaxf(*(shr_ptr + 4 + (tmp_loc + 2 & 3)), *(shr_ptr + 4 + (tmp_loc + 3 & 3)))) - fmaxf(fmaxf(*(shr_ptr + tmp_loc), *(shr_ptr + (tmp_loc + 1 & 3))), fmaxf(*(shr_ptr + (tmp_loc + 2 & 3)), *(shr_ptr + (tmp_loc + 3 & 3))));
				a_ptr -= 8;
			}
			ab_ptr -= half_abt_N;
			g_ptr--;
		}
		if (win_n > 0)
		{
			last_beta[blockIdx.x*half_abt_N + half_idx - 8] = ab;
		}
	}
}
__global__ void abKernel(float *dev_llr, float *dev_gamma, float *dev_ab, float *last_alfa, float *last_beta, int *dev_para, int iteration) //last iteration value
{
	__shared__ float shr_8illr[33 * 6 << 3];
	__shared__ float shr_8jllr[33 * 6 << 3];

	float plus, minus, gamma, ab;

	unsigned int half_idx = threadIdx.x%half_abt_N;
	unsigned int win_n = half_idx >> 3;
	unsigned int state_n = half_idx & 7;
	unsigned int i;
	int add_loc, sub_loc;
	float *g_ptr, *ab_ptr;

	//parameter set and pretrain
	if (threadIdx.x < half_abt_N)
	{
		/*get parameter from mem*/
		ab_ptr = dev_ab + ((blockIdx.x*info_L << 3) + half_idx);
		g_ptr = dev_gamma + (((blockIdx.x << 1) + dev_para[state_n])*info_tL + win_n*win_L);
		add_loc = dev_para[16 + state_n];
		sub_loc = dev_para[32 + state_n];
		/*get last alfa*/
		if (iteration == 0)
		{
			ab = state_n == 0 ? 0.0f : -10000.0f;

			if (win_n > 0)
			{
				for (i = 20; i >0; i--)
				{
					gamma = *(g_ptr - i);
					plus = __shfl(ab, add_loc, 8) + gamma;
					minus = __shfl(ab, sub_loc, 8) - gamma;
					ab = fmaxf(plus, minus);
				}
			}
		}
		else
		{
			ab = last_alfa[(blockIdx.x*half_abt_N) + half_idx];
		}
	}
	else
	{
		/*get parameter from mem*/
		ab_ptr = dev_ab + ((blockIdx.x*info_L << 3) + (win_L - 1)*half_abt_N + half_idx);
		g_ptr = dev_gamma + (((blockIdx.x << 1) + dev_para[8 + state_n])*info_tL + win_n*win_L + win_L - 1);
		add_loc = dev_para[24 + state_n];
		sub_loc = dev_para[40 + state_n];

		/*get last beta*/
		if (iteration == 0)
		{
			int v_L = win_n == win_N - 1 ? 3 : 20;
			ab = state_n == 0 ? 0.0f : -10000.0f;
			for (i = v_L; i >0; i--)
			{
				gamma = *(g_ptr + i);
				plus = __shfl(ab, add_loc, 8) + gamma;
				minus = __shfl(ab, sub_loc, 8) - gamma;
				ab = fmaxf(plus, minus);
			}
			if (win_n == win_N - 1)
			{
				last_beta[(blockIdx.x*half_abt_N) + half_idx] = ab;
			}
		}
		else
		{
			ab = last_beta[(blockIdx.x*half_abt_N) + half_idx];
		}
	}

	//calculate half alfa beta
	if (threadIdx.x < half_abt_N)
	{
		/*calculate half alfa*/
		for (i = 0; i < half_win_L; i++)
		{
			*ab_ptr = ab;
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = fmaxf(plus, minus);
			ab_ptr += half_abt_N;
			g_ptr++;

		}
	}
	else
	{
		/*calculate half beta*/
		for (i = 0; i < half_win_L; i++)
		{
			*ab_ptr = ab;
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = fmaxf(plus, minus);
			ab_ptr -= half_abt_N;
			g_ptr--;
		}
	}
	__syncthreads();
	/*calculate rest alfa\beta and prepare for llr*/

	if (threadIdx.x < half_abt_N)
	{
		float minus_tmp, plus_tmp;
		float *a_ptr = dev_llr + (blockIdx.x*info_L + win_n* win_L + half_win_L + state_n);
		float *shr_ptr = shr_8illr + ((win_n << 3) + (win_n >> 2) + state_n*33 * 6);
		//unsigned int tmp_loc = state_n & 3;

		for (int count = 0; i < win_L; i++, count++)
		{
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = fmaxf(plus, minus);

			minus = *ab_ptr + minus;
			plus = *ab_ptr + plus;

			minus_tmp = __shfl_down(minus, 4, 8);
			plus_tmp = __shfl_up(plus, 4, 8);
			if (state_n <4)
			{
				shr_8illr[count*33 * 6 + half_idx + (win_n >> 2)] = fmaxf(minus_tmp, minus);
			}
			else
			{
				shr_8illr[count*33 * 6 + half_idx + (win_n >> 2)] = fmaxf(plus_tmp, plus);
			}

			if (count == 7 || i == win_L - 1 && state_n<(half_win_L & 7))
			{
				count = -1;

				*a_ptr = fmaxf(fmaxf(*(shr_ptr + 7), *(shr_ptr + 6)), fmaxf(*(shr_ptr + 5), *(shr_ptr + 4 ))) - fmaxf(fmaxf(*(shr_ptr + 3), *(shr_ptr + 2)), fmaxf(*(shr_ptr + 1), *shr_ptr));
				a_ptr += 8;
			}
			ab_ptr += half_abt_N;
			g_ptr++;
		}
		if (win_n < win_N - 1)
		{
			last_alfa[blockIdx.x*half_abt_N + half_idx + 8] = ab;
		}
	}
	else
	{
		float minus_tmp, plus_tmp;
		float *a_ptr = dev_llr + (blockIdx.x*info_L + win_n* win_L + half_win_L - 1 - state_n);
		float *shr_ptr = shr_8jllr + ((win_n << 3) + (win_n >> 2) + state_n * 33 * 6);
		//unsigned int tmp_loc = state_n & 3;

		for (int count = 0; i < win_L; i++, count++)
		{
			gamma = *g_ptr;
			plus = __shfl(ab, add_loc, 8) + gamma;
			minus = __shfl(ab, sub_loc, 8) - gamma;
			ab = fmaxf(plus, minus);

			minus = *ab_ptr + minus;
			plus = *ab_ptr + plus;

			minus_tmp = __shfl_down(minus, 4, 8);
			plus_tmp = __shfl_up(plus, 4, 8);
			if (state_n <4)
			{
				shr_8jllr[count * 33 * 6 + half_idx + (win_n >> 2)] = fmaxf(minus_tmp, minus);
			}
			else
			{
				shr_8jllr[count * 33 * 6 + half_idx + (win_n >> 2)] = fmaxf(plus_tmp, plus);
			}

			if (count == 7 || i == win_L - 1 && state_n<(half_win_L & 7))
			{
				count = -1;

				*a_ptr = fmaxf(fmaxf(*(shr_ptr + 7), *(shr_ptr + 6)), fmaxf(*(shr_ptr + 5), *(shr_ptr + 4))) - fmaxf(fmaxf(*(shr_ptr + 3), *(shr_ptr + 2)), fmaxf(*(shr_ptr + 1), *shr_ptr));
				a_ptr -= 8;
			}
			ab_ptr -= half_abt_N;
			g_ptr--;
		}
		if (win_n > 0)
		{
			last_beta[blockIdx.x*half_abt_N + half_idx - 8] = ab;
		}
	}
}

__global__ void dataKernel(unsigned char *dev_data, float *dev_llr, int *dev_inter)
{
	__shared__ float shr_data[info_L];
	unsigned int i_idx, a_base, a_idx;
	
	a_idx = blockIdx.x*info_L;
	for (i_idx = threadIdx.x; i_idx < info_L; i_idx += info_thread_N)
	{
		shr_data[dev_inter[i_idx]] = dev_llr[a_idx + i_idx];	
	}
	__syncthreads();
	
	for (i_idx = threadIdx.x; i_idx < info_L; i_idx += info_thread_N)
	{
		dev_data[a_idx + i_idx] = shr_data[i_idx] > 0 ? 1 : 0;
		
	}
}
__global__ void data2Kernel(unsigned char *dev_data, float *dev_llr, int *dev_inter)
{
	__shared__ float shr_data[info_L];
	unsigned int i_idx, a_base, a_idx, i;
	i_idx = threadIdx.x;
	a_idx = blockIdx.x*info_L;
	for (i = 0; i < cal_time; i++)
	{
		shr_data[dev_inter[i_idx]] = dev_llr[a_idx + i_idx];
		i_idx += info_thread_N;
	}
	__syncthreads();
	for (i = 0; i < cal_time; i++)
	{
		i_idx -= info_thread_N;
		dev_data[a_idx + i_idx] = shr_data[i_idx] > 0 ? 1 : 0;

	}
}
/*
输出参数：
detected_data：		解码出的信息比特；

输入参数：
info_L：			信息比特长度；
input_c_fix：		输入Turbo码；
turbo_code_L：	输入的Turbo码长度；
interleaver_table：	交织表；
CQI					信道质量指示
*/
float cuTurboDecode(
	unsigned char  *detected_data,
	const float *input_c,
	const int *interleaver_table,
	int info_len,
	int turbo_code_L,
	int CQI)
{
	cudaEvent_t     start, stop;
	float duration;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*check device*/
	cudaDeviceProp  prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	if (!prop.deviceOverlap)
	{
		printf("Device will not handle overlaps, so no speed up from streams\n");
	}

	/*build mem for detected data in GPU*/
	unsigned char *dev_data;
	cudaMalloc((void**)&dev_data, block_N * info_len*sizeof(unsigned char));
	/*build mem for alfa and beta*/
	float *dev_ab;
	cudaMalloc((void**)&dev_ab, block_N * info_len * 8 * sizeof(float));//aaaa...bbb...aaa...bbb...
	/*gamma in device*/
	float *dev_gamma;
	cudaMalloc((void**)&dev_gamma, block_N * info_tL * 2 * sizeof(float));
	float *dev_a;
	cudaMalloc((void**)&dev_a, block_N * info_len * sizeof(float));

	float *dev_llr;
	cudaMalloc((void**)&dev_llr, block_N * info_len  * sizeof(float));

	float *last_alfa0, *last_beta0, *last_alfa1, *last_beta1;
	cudaMalloc((void**)&last_alfa0, block_N * win_N * 8 * sizeof(float));
	cudaMalloc((void**)&last_beta0, block_N * win_N * 8 * sizeof(float));
	cudaMalloc((void**)&last_alfa1, block_N * win_N * 8 * sizeof(float));
	cudaMalloc((void**)&last_beta1, block_N * win_N * 8 * sizeof(float));
	float last_init[8] = { 0, -100, -100, -100, -100, -100, -100, -100 };
	for (int i = 0; i < block_N; i++)
	{
		cudaMemcpy(last_alfa0 + i * win_N * 8, last_init, 8 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(last_alfa1 + i * win_N * 8, last_init, 8 * sizeof(float), cudaMemcpyHostToDevice);
	}

	/*copy interleave table from host to device*/
	int *dev_inter;
	cudaMalloc((void**)&dev_inter, info_len*sizeof(int));
	cudaMemcpy(dev_inter, interleaver_table, info_len*sizeof(int), cudaMemcpyHostToDevice);

	int *dev_para;
	cudaMalloc((void**)&dev_para, 48 * sizeof(int));
	int tmp[48] = { 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1,
		1, 2, 5, 6, 0, 3, 4, 7, 4, 0, 1, 5, 6, 2, 3, 7,
		0, 3, 4, 7, 1, 2, 5, 6, 0, 4, 5, 1, 2, 6, 7, 3 };
	cudaMemcpy(dev_para, tmp, 48 * sizeof(int), cudaMemcpyHostToDevice);

	float *dev_s0;
	float *dev_p0;
	float *dev_s1;
	float *dev_p1;

	cudaMalloc((void**)&dev_s0, block_N*info_tL*sizeof(float));
	cudaMalloc((void**)&dev_p0, block_N*info_tL*sizeof(float));
	cudaMalloc((void**)&dev_s1, block_N*info_tL*sizeof(float));
	cudaMalloc((void**)&dev_p1, block_N*info_tL*sizeof(float));

	/*copy s/p from host to device*/
	float *upper_s = new float[info_tL];
	float *upper_p = new float[info_tL];
	float *lower_s = new float[info_tL];
	float *lower_p = new float[info_tL];
	switch (CQI)
	{
	case 0:
	case 1:
	case 2:
	case 3:
	case 12:
	{
		depuncture1(upper_s, upper_p, lower_s, lower_p, input_c, info_len, turbo_code_L);
		break;
	}
	case 4:
	case 6:
	{
		depuncture2(upper_s, upper_p, lower_s, lower_p, input_c, info_len, turbo_code_L);
		break;
	}
	case 5:
	case 7:
	case 9:
	case 10:
	case 11:
	{
		depuncture3(upper_s, upper_p, lower_s, lower_p, input_c, info_len, turbo_code_L);
		break;
	}
	case 8:
	{
		depuncture4(upper_s, upper_p, lower_s, lower_p, input_c, info_len, turbo_code_L);
		break;
	}
	default:
	{
		printf("CQI must between 0 and 11\n");
	}
	}
	interleaveOut(lower_s, upper_s, interleaver_table, info_len);

	for (int i = 0; i < block_N; i++)
	{
		cudaMemcpy(dev_s0 + i*info_tL, upper_s, info_tL*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_p0 + i*info_tL, upper_p, info_tL*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_s1 + i*info_tL, lower_s, info_tL*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_p1 + i*info_tL, lower_p, info_tL*sizeof(float), cudaMemcpyHostToDevice);
	}
	delete[] upper_p;
	delete[] upper_s;
	delete[] lower_p;
	delete[] lower_s;

	/*use one smx to handle 2 blocks each has N threads*/

	cudaEventRecord(start, 0);			 //event record

	gammaKernel << <block_N, info_thread_tN >> > (dev_gamma, dev_s0, dev_p0, NULL);
	for (int iter = 0; iter < 6; iter++)
	{

		abKernel << <block_N, ab_thread_N >> >(dev_llr, dev_gamma, dev_ab, last_alfa0, last_beta0, dev_para, iter);

		extKernel << <block_N, info_thread_N >> >(dev_llr, dev_a, dev_s0, dev_inter, 0);

		gammaKernel << <block_N, info_thread_tN >> >(dev_gamma, dev_s1, dev_p1, dev_a);


		abKernel << <block_N, ab_thread_N >> >(dev_llr, dev_gamma, dev_ab, last_alfa1, last_beta1, dev_para, iter);
		if (iter < 5)
		{
			extKernel << <block_N, info_thread_N >> >(dev_llr, dev_a, dev_s1, dev_inter, 1);
			
			gammaKernel << <block_N, info_thread_tN >> > (dev_gamma, dev_s0, dev_p0, dev_a);
		}
	}
	dataKernel << <block_N, info_thread_N >> >(dev_data, dev_llr, dev_inter);

	cudaEventRecord(stop, 0);				 //event record
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);

	cudaMemcpy(detected_data, dev_data, info_len*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_s0);
	cudaFree(dev_p0);
	cudaFree(last_alfa0);
	cudaFree(last_alfa1);
	cudaFree(last_beta0);
	cudaFree(last_beta1);
	cudaFree(dev_s1);
	cudaFree(dev_p1);
	cudaFree(dev_data);
	cudaFree(dev_ab);
	cudaFree(dev_gamma);
	cudaFree(dev_a);
	cudaFree(dev_llr);
	//(cudaUnbindTexture(tex_inter);
	//(cudaUnbindTexture(tex_para);
	cudaFree(dev_inter);
	cudaFree(dev_para);

	return duration;
}
