/*
 * File:   AwgnNoise.h
 *
 * Brief:  AwgnNoise() is used to add AWGN noise to the tansmitted symbols 
 *         according to the setted SNR(dB).
 *
 * Author: Sun Cai
 *
 * Last mod file time: 2015/4/7
 *
 */

#ifndef _AWGNNOISE_H_
#define _AWGNNOISE_H_

/*******************************************************************************
**							INCLUDE FILES
*******************************************************************************/
#include <math.h> //pow(), sqrt(), log(), cos().
#include <stdlib.h> // srand(), rand(), RAND_MAX.

/*******************************************************************************
**                       INTERNAL MACRO DEFINITIONS
*******************************************************************************/
#define Pi 3.141593f

/******************************************************************************
**                     EXTERNAL VARIABLE DECLARATIONS
*******************************************************************************/

/******************************************************************************
**                      EXTERNAL FUNCTION PROTOTYPES
*******************************************************************************/
void AwgnNoise(float sigma2, int length_in, float *FadingSignal_real, float *FadingSignal_img);

#endif
/***************************** End Of File ************************************/
