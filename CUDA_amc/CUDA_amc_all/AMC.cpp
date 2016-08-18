//Perform adaptive modulation and coding.
//(c) 2015 by NCRL

/* Parameters:
CQI                            CQI
N_Info                         Number of information bits
N_RI						   Number of streams (Rank) of MIMO
Qm							   Modulation order
N_RB						   Number of resource block (N_RB = 6)
N_CB						   Number of code block
N_RE_PerRB					   Number of resource elements per RB (N_RE_PerRB = 150)
pun_pattern					   puncture pattern for data bits in turbo code
inner_code_interleaver		   interleaver used by turbo code
outer_code_interleaver		   interleaver used after turbo code
rate_matching_vector		   index of the bits that should be punctured for rate matching
de_rate_matching_vector		   index of the information bits
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* library of functions */
#include "AMC.h"
#include "convolutional.h"
#include "maxstar.h"
#include "siso.h"
#include "crc.h"
#include "AwgnNoise.h"
#include "Turbo.h"
#include "Data.h"
/*library of GPU*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
//#include "../common/book.h"
static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define N_RB 6 
#define N_RE_PerRB 150
int CQI_test(int CQI, double snr0, double snr1, double snr_step)
{

	/* Parameters Definition */
	int *pun_pattern = NULL, *inner_code_interleaver = NULL, *outer_code_interleaver = NULL, *rate_matching_vector = NULL, *de_rate_matching_vector = NULL;
	int N_CB, N_Info, N_RI, Qm, pun_period, CB_length;
	int i, j, k, n, temp_count, count_period;
	float code_rate, sigma2;

	/* CQI Mapping */
	//CQI = 2;
	float abs_mean_llr = 0;

	switch (CQI)
	{
	case 0:
		pun_period = 4;
		pun_pattern = pun_pattern_4;
		inner_code_interleaver = interleaver600;
		outer_code_interleaver = interleaver1800;
		rate_matching_vector = de_rate_matching_vector_CQI_0;
		de_rate_matching_vector = de_rate_matching_vector_CQI_0;
		Qm = 2;
		N_CB = 1;
		N_RI = 1;
		N_Info = 576;
		CB_length = 1800;
		code_rate = 1.0 / 3;
		break;

	case 1:
		pun_period = 4;
		pun_pattern = pun_pattern_4;
		inner_code_interleaver = interleaver1200;
		outer_code_interleaver = interleaver3600;
		rate_matching_vector = de_rate_matching_vector_CQI_19;
		de_rate_matching_vector = de_rate_matching_vector_CQI_19;
		Qm = 2;
		N_CB = 1;
		N_RI = 2;
		N_Info = 1176;
		CB_length = 3600;
		code_rate = 1.0 / 3;
		break;

	case 2:
		pun_period = 4;
		pun_pattern = pun_pattern_4;
		inner_code_interleaver = interleaver1800;
		outer_code_interleaver = interleaver5400;
		rate_matching_vector = rate_matching_vector_CQI_211;
		de_rate_matching_vector = de_rate_matching_vector_CQI_211;
		Qm = 2;
		N_CB = 1;
		N_RI = 3;
		N_Info = 1776;
		CB_length = 5400;
		code_rate = 1.0 / 3;
		break;

	case 3:
		pun_period = 4;
		pun_pattern = pun_pattern_4;
		inner_code_interleaver = interleaver400;
		outer_code_interleaver = interleaver7200;
		rate_matching_vector = rate_matching_vector_CQI_345;
		de_rate_matching_vector = de_rate_matching_vector_CQI_345;
		Qm = 2;
		N_CB = 6;
		N_RI = 4;
		N_Info = 376;
		CB_length = 1200;
		code_rate = 1.0 / 3;
		break;

	case 4:
		pun_period = 8;
		pun_pattern = pun_pattern_8;
		inner_code_interleaver = interleaver600;
		outer_code_interleaver = interleaver7200;
		rate_matching_vector = rate_matching_vector_CQI_345;
		de_rate_matching_vector = de_rate_matching_vector_CQI_345;
		Qm = 2;
		N_CB = 6;
		N_RI = 4;
		N_Info = 576;
		CB_length = 1200;
		code_rate = 1.0 / 2;
		break;

	case 5:
		pun_period = 16;
		pun_pattern = pun_pattern_16;
		inner_code_interleaver = interleaver800;
		outer_code_interleaver = interleaver7200;
		rate_matching_vector = rate_matching_vector_CQI_345;
		de_rate_matching_vector = de_rate_matching_vector_CQI_345;
		Qm = 2;
		N_CB = 6;
		N_RI = 4;
		N_Info = 776;
		CB_length = 1200;
		code_rate = 2.0 / 3;
		break;

	case 6:
		pun_period = 8;
		pun_pattern = pun_pattern_8;
		inner_code_interleaver = interleaver1200;
		outer_code_interleaver = interleaver14400;
		rate_matching_vector = rate_matching_vector_CQI_678;
		de_rate_matching_vector = de_rate_matching_vector_CQI_678;
		Qm = 4;
		N_CB = 6;
		N_RI = 4;
		N_Info = 1176;
		CB_length = 2400;
		code_rate = 1.0 / 2;
		break;

	case 7:
		pun_period = 16;
		pun_pattern = pun_pattern_16;
		inner_code_interleaver = interleaver1600;
		outer_code_interleaver = interleaver14400;
		rate_matching_vector = rate_matching_vector_CQI_678;
		de_rate_matching_vector = de_rate_matching_vector_CQI_678;
		Qm = 4;
		N_CB = 6;
		N_RI = 4;
		N_Info = 1576;
		CB_length = 2400;
		code_rate = 2.0 / 3;
		break;

	case 8:
		pun_period = 24;
		pun_pattern = pun_pattern_24;
		inner_code_interleaver = interleaver1800;
		outer_code_interleaver = interleaver14400;
		rate_matching_vector = rate_matching_vector_CQI_678;
		de_rate_matching_vector = de_rate_matching_vector_CQI_678;
		Qm = 4;
		N_CB = 6;
		N_RI = 4;
		N_Info = 1776;
		CB_length = 2400;
		code_rate = 3.0 / 4;
		break;

	case 9:
		pun_period = 16;
		pun_pattern = pun_pattern_16;
		inner_code_interleaver = interleaver2400;
		outer_code_interleaver = interleaver21600;
		rate_matching_vector = rate_matching_vector_CQI_19;
		de_rate_matching_vector = de_rate_matching_vector_CQI_19;
		Qm = 6;
		N_CB = 6;
		N_RI = 4;
		N_Info = 2376;
		CB_length = 3600;
		code_rate = 2.0 / 3;
		break;

	case 10:
		pun_period = 16;
		pun_pattern = pun_pattern_16;
		inner_code_interleaver = interleaver3000;
		outer_code_interleaver = interleaver27000;
		rate_matching_vector = rate_matching_vector_CQI_10;
		de_rate_matching_vector = de_rate_matching_vector_CQI_10;
		Qm = 6;
		N_CB = 6;
		N_RI = 5;
		N_Info = 2976;
		CB_length = 4500;
		code_rate = 2.0 / 3;
		break;

	case 11:
		pun_period = 16;
		pun_pattern = pun_pattern_16;
		inner_code_interleaver = interleaver3600;
		outer_code_interleaver = interleaver32400;
		rate_matching_vector = rate_matching_vector_CQI_211;
		de_rate_matching_vector = de_rate_matching_vector_CQI_211;
		Qm = 6;
		N_CB = 6;
		N_RI = 6;
		N_Info = 3576;
		CB_length = 5400;
		code_rate = 2.0 / 3;
		break;

	default:;
	}

	/* Encode */
	int DataLength = N_Info + 24;
	int CodeLength = 2 * (DataLength + 3);

	int *a = new int[DataLength];
	int *b = new int[CodeLength];
	int *c = new int[N_RE_PerRB * N_RI * Qm * N_RB];
	float *d = new float[N_RE_PerRB * N_RI * Qm * N_RB]();
	float *e = new float[CB_length + 12]();

	unsigned char *a_char = new unsigned char[DataLength];
	unsigned char *data = new unsigned char[N_Info * N_CB];
	unsigned char *detected_data = new unsigned char[DataLength];

	float *de_unpunctured = new float[2 * CodeLength]();
	int *unpunctured = new int[2 * CodeLength];
	int *punctured = new int[CB_length + 12];

	float *Sig_I = new float[N_RE_PerRB * N_RI * N_RB];
	float *Sig_Q = new float[N_RE_PerRB * N_RI * N_RB];

	//cudaEvent_t     start, stop;
	//HANDLE_ERROR(cudaEventCreate(&start));
	//HANDLE_ERROR(cudaEventCreate(&stop));
	float duration;
	float total_time;
	const int streamNum = 1;
	const int multi_frame = 10;// 0000000 / (streamNum*N_CB * N_Info) + 1;
	int error;
	//char name[100];
	//sprintf(name, "scale%.1f.txt", scale);
	//FILE *fp = fopen("best_noLc_15.txt", "a+");
	//fprintf(fp, "scale%.1f\n", scale);
	for (double nSNR = snr0; nSNR <= snr1; nSNR += snr_step)
	{

		total_time = 0;
		error = 0;
		for (int frame = 0; frame < streamNum * multi_frame; frame++)
		{
			/* generate a subframe */
			srand(time(0));
			for (n = 0; n < N_CB * N_Info; n++)
			{
				data[n] = (unsigned char)(rand() % 2);
			}

			for (n = 0; n < N_CB; n++)
			{

				for (i = 0; i < N_Info; i++)
				{
					a_char[i] = data[i + n * N_Info];
				}

				/* CRC */
				tx_append_crc(a_char, N_Info);
				for (i = 0; i < DataLength; i++)
				{
					a[i] = (int)a_char[i];
				}

				/* Turbo coding */
				/* upper */
				conv_encode(b, a, out0, state0, out1, state1, tail, 4, DataLength, 2);
				for (i = 0; i < CodeLength / 2; i++)
				{
					for (j = 0; j < 2; j++)
					{
						unpunctured[i * 4 + j] = b[i * 2 + j];
					}
				}

				/* lower */
				int *temp = new int[DataLength];
				for (i = 0; i < DataLength; i++)
				{
					temp[i] = a[i];
				}
				for (i = 0; i < DataLength; i++)
				{
					a[i] = temp[inner_code_interleaver[i]];
				}

				conv_encode(b, a, out0, state0, out1, state1, tail, 4, DataLength, 2);

				for (i = 0; i < CodeLength / 2; i++)
				{
					for (j = 0; j < 2; j++)
					{
						unpunctured[i * 4 + 2 + j] = b[i * 2 + j];
					}
				}

				/* puncture */
				k = 0;

				/* data */
				count_period = 0;
				for (i = 0; i < 4 * DataLength; i++)
				{
					temp_count = pun_pattern[count_period];
					while (temp_count > 0)
					{
						punctured[k] = unpunctured[i];
						temp_count--;
						k++;
					}
					count_period++;
					if (count_period == pun_period)
						count_period = 0;
				}

				/* tail */
				for (i = 0; i < 3; i++)
				{
					for (j = 0; j < 2; j++)
					{
						punctured[k] = unpunctured[4 * DataLength + i * 4 + j];
						k++;
					}
				}
				for (i = 0; i < 3; i++)
				{
					for (j = 0; j < 2; j++)
					{
						punctured[k] = unpunctured[4 * DataLength + i * 4 + j + 2];
						k++;
					}
				}

				/* rate matching */
				for (i = 0; i < CB_length; i++)
				{
					c[i * N_CB + n] = punctured[de_rate_matching_vector[i]];
				}
				delete[] temp;
			}


			/* inter-RB interleaver */
			int *temp = new int[N_RE_PerRB * N_RI * Qm * N_RB];
			for (i = 0; i < N_RE_PerRB * N_RI * Qm * N_RB; i++)
			{
				temp[i] = c[i];
			}
			for (i = 0; i < N_RE_PerRB * N_RI * Qm * N_RB; i++)
			{
				c[i] = temp[outer_code_interleaver[i]];
			}

			delete[] temp;
			/* modulation 	*/
			QAM_Modulation(c, Sig_I, Sig_Q, Qm, N_RE_PerRB * N_RI * N_RB);

			/* channel */
			sigma2 = sqrt(1.0 / (pow(10.0, (nSNR / 10.0)) * code_rate * Qm * 2.0));
			AwgnNoise(sigma2, N_RE_PerRB * N_RI * N_RB, Sig_I, Sig_Q);

			/* demodulation*/
			QAM_Demodulation(d, Sig_I, Sig_Q, 2 * pow(sigma2, 2), N_RE_PerRB * N_RB, Qm, N_RI);

			/* Decode */

			/* inter-RB deinterleaver */
			/*for(i = 0; i < N_RE_PerRB * N_RI * Qm * N_RB; i++)
			{
			if(c[i] == 0)
			d[i] = -3;
			else
			d[i] = 3;
			}*/



			float *temp_float = new float[N_RE_PerRB * N_RI * Qm * N_RB];
			for (i = 0; i < N_RE_PerRB * N_RI * Qm * N_RB; i++)
			{
				temp_float[i] = d[i];
			}
			for (i = 0; i < N_RE_PerRB * N_RI * Qm * N_RB; i++)
			{
				d[outer_code_interleaver[i]] = temp_float[i];
			}
			delete[] temp_float;

			/* turbo decoding */
			for (n = 0; n < N_CB; n++)
			{
				for (i = 0; i < CB_length; i++)
				{
					e[de_rate_matching_vector[i]] = d[i * N_CB + n];
				}

				duration = cuTurboDecode(detected_data, e, inner_code_interleaver, DataLength, CB_length + 12, CQI);

				total_time += duration;
				for (i = 0; i < N_Info; i++)
				{
					error += abs(detected_data[i] - data[N_Info*n + i]);
				}
			}
		}

		total_time = total_time / multi_frame;
		printf("%fdB  %fms  %fmbps  %f\n", nSNR, total_time, N_CB*streamNum*2.4 / total_time, (double)error / (N_CB * N_Info * streamNum) / multi_frame);
		//fprintf(fp, "%f\n",(double)error / (N_CB * N_Info * streamNum) / multi_frame);
	}
	printf("\n");
	//fprintf(fp, "\n");
	//fprintf(fp, "\n");
	//fprintf(fp, "\n");
	//fprintf(fp, "\n");
	//fclose(fp);
	/* free the dynamically allocated memory */
	delete[] unpunctured;
	delete[] punctured;
	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
	delete[] e;
	delete[] de_unpunctured;
	delete[] detected_data;
	delete[] Sig_I;
	delete[] Sig_Q;
	delete[] a_char;
	delete[] data;

	//HANDLE_ERROR(cudaEventDestroy(start));
	//HANDLE_ERROR(cudaEventDestroy(stop));
	return 0;
}

int main()
{
	CQI_test(9, 5, 10, 0.5);
	//CQI_test(0, 0, 2, 0.2, scale[0]);
	//CQI_test(1, 0, 2, 0.2, scale[1]);
	//CQI_test(2, 0, 2, 0.2, scale[2]);
	//CQI_test(3, 0, 3, 0.3, scale[3]);
	//CQI_test(4, 0, 3, 0.3, scale[4]);
	//CQI_test(5, 0, 5, 0.5, scale[5]);
	//CQI_test(6, 1, 6, 0.5, scale[6]);
	//CQI_test(7, 2, 7, 0.5, scale[7]);
	//CQI_test(8, 3, 8, 0.5, scale[8]);
	//CQI_test(9, 5, 10, 0.5, scale[9]);
	//CQI_test(10, 5, 10, 0.5, scale[10]);
	//CQI_test(11, 5, 10, 0.5, scale[11]);

	//////	
	//CQI_test(0, 15, 50, 5, scale[0]);
	//CQI_test(1, 15, 50, 5, scale[1]);
	//CQI_test(2, 15, 50, 5, scale[2]);
	//CQI_test(3, 15, 50, 5, scale[3]);
	//CQI_test(4, 15, 50, 5, scale[4]);
	//CQI_test(5, 15, 50, 5, scale[5]);
	//CQI_test(6, 15, 50, 5, scale[6]);
	//CQI_test(7, 15, 50, 5, scale[7]);
	//CQI_test(8, 15, 50, 5, scale[8]);
	//CQI_test(9, 15, 50, 5, scale[9]);
	//CQI_test(10, 15, 50, 5, scale[10]);
	//CQI_test(11, 15, 50, 5, scale[11]);
	getchar();
}

