/*
 * File:   AwgnNoise.c
 *
 * Brief:  random_uniform() is to generate a number with (0,1) uniform distribution.
 *         rand_normal() is to generate two independent numbers with standard normal
 *         distribution using Box-muller method.
 *
 * Author: Sun Cai
 *
 * Last mod file time: 2015/3/25 
 *
 */
#include "AwgnNoise.h"
#include <iostream>
 
/*******************************************************************************
**                      INTERNAL FUNCTION PROTOTYPES
*******************************************************************************/

float random_uniform();
void rand_normal(float *n_real, float *n_img);
void AwgnNoise(float sigma2, int length_in, float *FadingSignal_real, float *FadingSignal_img);

/*******************************************************************************
**                      INTERNAL VARIABLE DEFINITIONS
*******************************************************************************/
unsigned noise[3] = {173, 173, 173};
/*******************************************************************************
**                     FUNCTION DEFINITIONS
*******************************************************************************/
void AwgnNoise(float sigma2, int length_in, float *FadingSignal_real, float *FadingSignal_img)
{
	int i;
    
    float tmp_noise_real, tmp_noise_img;
	float noise_real[9222];
	float noise_img[9222];

	/****************Generate random number**************/
	for(i = 0; i < length_in; i++)
	{
		/* the real and imaginay part of the noise are independent */
        rand_normal(&tmp_noise_real, &tmp_noise_img);
		noise_real[i] = sigma2 * tmp_noise_real;
		noise_img[i] = sigma2 * tmp_noise_img;
		//cout << "noise_real[" << i << "] =" << noise_real[i];
		//cout << "noise_img[" << i << "] =" << noise_img[i] <<endl;
	}
	/****************output**************/
	for(i = 0; i < length_in; i++)
	{
		FadingSignal_real[i] = FadingSignal_real[i] + noise_real[i];
		FadingSignal_img[i] = FadingSignal_img[i] + noise_img[i];
	}
}


void rand_normal(float *n_real, float *n_img)
{
	float u, v;
    	float R, theta;
    
	u = random_uniform();
	v = random_uniform();
    	R = sqrt( (-2.0) * log(v) );
    	theta = 2.0*Pi*u;
	*n_real = R * cos( theta );
    	*n_img = R * sin( theta );
    
}

float random_uniform()
{
	float u = 0.0;
    
	noise[0] = (noise[0]*249) % 61967;
	noise[1] = (noise[1]*251) % 63443;
	noise[2] = (noise[2]*252) % 63599;

	u = (float)((noise[0]/61967.0 + noise[1]/63443.0 + noise[2]/63599.0)
		- (int)(noise[0]/61967.0 + noise[1]/63443.0 + noise[2]/63599.0));

	return u;
}

/*
float random_uniform()
{    
	return rand() % 2;
}*/ 

/* According to the time-based seed which was initialised by srand(), a random number is created.
 * If no seed was initialised by srand(), rand() would use the seed initialised by srand(1) automatically
 * and a pseudo-random number would be created. 1. DO NOT initialise time-based seed within this sub-function,
 * because everytime when this function is called, the seed would be initialised newly. CPU runs repaidly, 
 * there may be no time-difference among several successive calling for the function, and the seeds initialised
 * are more likely the same, so the phenomenon that the random numbers obtained are all the same may occur.
 * 2. DO NOT create other seed using srand() when trying to get independent random numbers, e.g., using 
 * srand(loop_index), because you must get independent seeds to ensure independence among the generated numbers.
 * 3. Although the seed initialised by srand(1) is automatically used by rand(), the result of Debug indicates 
 * that rand() generates different even nearly uncorrelated random numbers in every loop!
*/
/***************************** End Of File ************************************/
