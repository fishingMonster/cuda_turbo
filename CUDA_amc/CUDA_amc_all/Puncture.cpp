#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include "Turbo.h"
 
void depuncture1(float *input_upper_s ,float *input_upper_p ,float *input_lower_s ,float *input_lower_p ,const float *input_c_fix, int info_len, int turbo_code_len)
{
	if (turbo_code_len - 3 * info_len - 12)
	{
		printf("code length error\n");
	}
	/* depuncture the coded data bits  */
	int p;
	for (p = 0; p < info_len; p++)
	{
		*(input_upper_s + p) = *(input_c_fix + 3 * p);
		*(input_upper_p + p) = *(input_c_fix + 3 * p + 1);
		*(input_lower_p + p) = *(input_c_fix + 3 * p + 2);
	}

	/* depuncture the coded tail bits, if height = 4, then the tail bits are rearranged to conform to UMTS convention */

}
void depuncture2(float *input_upper_s, float *input_upper_p, float *input_lower_s, float *input_lower_p, const float *input_c_fix, int info_len, int turbo_code_len)
{
	if (turbo_code_len - 2 * info_len - 12)
	{
		printf("code length error\n");
	}

	/* depuncture the coded data bits  */
	int p;
	for (p = 0; p < info_len / 2; p++)
	{
		*(input_upper_s + 2 * p) = *(input_c_fix + 4 * p);
		*(input_upper_s + 2 * p + 1) = *(input_c_fix + 4 * p + 2);
		*(input_upper_p + 2 * p) = *(input_c_fix + 4 * p + 1);
		*(input_upper_p + 2 * p + 1) = 0;
		*(input_lower_p + 2 * p) = 0;
		*(input_lower_p + 2 * p + 1) = *(input_c_fix + 4 * p + 3);
	}

	/* depuncture the coded tail bits, if height = 4, then the tail bits are rearranged to conform to UMTS convention */
	*(input_upper_s + info_len) = *(input_c_fix + turbo_code_len - 12);
	*(input_upper_s + info_len + 1) = *(input_c_fix + turbo_code_len - 10);
	*(input_upper_s + info_len + 2) = *(input_c_fix + turbo_code_len - 8);
	*(input_upper_p + info_len) = *(input_c_fix + turbo_code_len - 11);
	*(input_upper_p + info_len + 1) = *(input_c_fix + turbo_code_len - 9);
	*(input_upper_p + info_len + 2) = *(input_c_fix + turbo_code_len - 7);

	*(input_lower_s + info_len) = *(input_c_fix + turbo_code_len - 6);
	*(input_lower_s + info_len + 1) = *(input_c_fix + turbo_code_len - 4);
	*(input_lower_s + info_len + 2) = *(input_c_fix + turbo_code_len - 2);
	*(input_lower_p + info_len) = *(input_c_fix + turbo_code_len - 5);
	*(input_lower_p + info_len + 1) = *(input_c_fix + turbo_code_len - 3);
	*(input_lower_p + info_len + 2) = *(input_c_fix + turbo_code_len - 1);
}
void depuncture3(float *input_upper_s, float *input_upper_p, float *input_lower_s, float *input_lower_p, const float *input_c_fix, int info_len, int turbo_code_len)
{
	if (turbo_code_len - info_len / 2 * 3 - 12)
	{
		printf("code length error\n");
	}
	/* depuncture the coded data bits  */
	int p;
	for (p = 0; p < info_len / 4; p++)
	{
		*(input_upper_s + 4 * p) = *(input_c_fix + 6 * p);
		*(input_upper_s + 4 * p + 1) = *(input_c_fix + 6 * p + 2);
		*(input_upper_s + 4 * p + 2) = *(input_c_fix + 6 * p + 3);
		*(input_upper_s + 4 * p + 3) = *(input_c_fix + 6 * p + 5);
		*(input_upper_p + 4 * p) = *(input_c_fix + 6 * p + 1);
		*(input_upper_p + 4 * p + 1) = 0;
		*(input_upper_p + 4 * p + 2) = 0;
		*(input_upper_p + 4 * p + 3) = 0;
		*(input_lower_p + 4 * p) = 0;
		*(input_lower_p + 4 * p + 1) = 0;
		*(input_lower_p + 4 * p + 2) = *(input_c_fix + 6 * p + 4);
		*(input_lower_p + 4 * p + 3) = 0;
	}

	/* depuncture the coded tail bits, if height = 4, then the tail bits are rearranged to conform to UMTS convention */
	*(input_upper_s + info_len) = *(input_c_fix + turbo_code_len - 12);
	*(input_upper_s + info_len + 1) = *(input_c_fix + turbo_code_len - 10);
	*(input_upper_s + info_len + 2) = *(input_c_fix + turbo_code_len - 8);
	*(input_upper_p + info_len) = *(input_c_fix + turbo_code_len - 11);
	*(input_upper_p + info_len + 1) = *(input_c_fix + turbo_code_len - 9);
	*(input_upper_p + info_len + 2) = *(input_c_fix + turbo_code_len - 7);

	*(input_lower_s + info_len) = *(input_c_fix + turbo_code_len - 6);
	*(input_lower_s + info_len + 1) = *(input_c_fix + turbo_code_len - 4);
	*(input_lower_s + info_len + 2) = *(input_c_fix + turbo_code_len - 2);
	*(input_lower_p + info_len) = *(input_c_fix + turbo_code_len - 5);
	*(input_lower_p + info_len + 1) = *(input_c_fix + turbo_code_len - 3);
	*(input_lower_p + info_len + 2) = *(input_c_fix + turbo_code_len - 1);
}
void depuncture4(float *input_upper_s, float *input_upper_p, float *input_lower_s, float *input_lower_p, const float *input_c_fix, int info_len, int turbo_code_len)
{
	if (turbo_code_len - info_len / 3 * 4 - 12)
	{
		printf("code length error\n");
	}
	/* depuncture the coded data bits  */
	int p;
	for (p = 0; p < info_len / 6; p++)
	{
		*(input_upper_s + 6 * p) = *(input_c_fix + 8 * p);
		*(input_upper_s + 6 * p + 1) = *(input_c_fix + 8 * p + 2);
		*(input_upper_s + 6 * p + 2) = *(input_c_fix + 8 * p + 3);
		*(input_upper_s + 6 * p + 3) = *(input_c_fix + 8 * p + 4);
		*(input_upper_s + 6 * p + 4) = *(input_c_fix + 8 * p + 6);
		*(input_upper_s + 6 * p + 5) = *(input_c_fix + 8 * p + 7);
		*(input_upper_p + 6 * p) = *(input_c_fix + 8 * p + 1);
		*(input_upper_p + 6 * p + 1) = 0;
		*(input_upper_p + 6 * p + 2) = 0;
		*(input_upper_p + 6 * p + 3) = 0;
		*(input_upper_p + 6 * p + 4) = 0;
		*(input_upper_p + 6 * p + 5) = 0;
		*(input_lower_p + 6 * p) = 0;
		*(input_lower_p + 6 * p + 1) = 0;
		*(input_lower_p + 6 * p + 2) = 0;
		*(input_lower_p + 6 * p + 3) = *(input_c_fix + 8 * p + 5);
		*(input_lower_p + 6 * p + 4) = 0;
		*(input_lower_p + 6 * p + 5) = 0;
	}

	/* depuncture the coded tail bits, if height = 4, then the tail bits are rearranged to conform to UMTS convention */
	*(input_upper_s + info_len) = *(input_c_fix + turbo_code_len - 12);
	*(input_upper_s + info_len + 1) = *(input_c_fix + turbo_code_len - 10);
	*(input_upper_s + info_len + 2) = *(input_c_fix + turbo_code_len - 8);
	*(input_upper_p + info_len) = *(input_c_fix + turbo_code_len - 11);
	*(input_upper_p + info_len + 1) = *(input_c_fix + turbo_code_len - 9);
	*(input_upper_p + info_len + 2) = *(input_c_fix + turbo_code_len - 7);

	*(input_lower_s + info_len) = *(input_c_fix + turbo_code_len - 6);
	*(input_lower_s + info_len + 1) = *(input_c_fix + turbo_code_len - 4);
	*(input_lower_s + info_len + 2) = *(input_c_fix + turbo_code_len - 2);
	*(input_lower_p + info_len) = *(input_c_fix + turbo_code_len - 5);
	*(input_lower_p + info_len + 1) = *(input_c_fix + turbo_code_len - 3);
	*(input_lower_p + info_len + 2) = *(input_c_fix + turbo_code_len - 1);
}

