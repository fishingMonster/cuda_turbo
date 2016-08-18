#ifndef _TURBOCODE_H_
#define _TURBOCODE_H_

typedef struct
{
	float *gamma;
	float *ab;

} cudaMap;

typedef struct
{
	float *s;
	float *p;
}cudaDec;

float gaussrand(void);
//void interleave(float *msg, int length, const int *interleave_table);
void deinterleave(float *msg, const int *interleave_table, int length);
void deinterleaveOut(float *output, const float *input, const int *interleaver_table, int length);
void interleaveOut(float *output, const float *input, const int *interleave_table, int length);//store interleaved data in another Pattern
void rsc_encode(float *msg, float *parity, int info_len, int info_tail_len);

void depuncture1(float *input_upper_s, float *input_upper_p, float *input_lower_s, float *input_lower_p, const float *input_c_fix, int info_len, int turbo_code_len);
void depuncture2(float *input_upper_s, float *input_upper_p, float *input_lower_s, float *input_lower_p, const float *input_c_fix, int info_len, int turbo_code_len);
void depuncture3(float *input_upper_s, float *input_upper_p, float *input_lower_s, float *input_lower_p, const float *input_c_fix, int info_len, int turbo_code_len);
void depuncture4(float *input_upper_s, float *input_upper_p, float *input_lower_s, float *input_lower_p, const float *input_c_fix, int info_len, int turbo_code_len);
void float2Limitfloat(float *output, const float *input, float	scale, int length);
void CRC24(float *crc24, const float *info_bit, int length);
float cuTurboDecode(
	unsigned char  *detected_data,
	const float *input_c,
	const int *interleaver_table,
	int info_L,
	int turbo_code_L,
	int CQI);
#endif 