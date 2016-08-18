/*
 * crc_append_check.cpp
 *
 *  Created on: Jul 10, 2015
 *      Author: leo
 */
#include <iostream>
#include <cstddef>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include "crc.h"

 

//append crc
void tx_append_crc(unsigned char* data, int input_data_length) 
{
	//bit2byte
	unsigned char* data_bytes = new unsigned char[input_data_length / 8];
	bit2byte(data, data_bytes, input_data_length);
	
	//crc_byte
	unsigned char* crc_bytes = new unsigned char[3];
	crc(data_bytes, crc_bytes, input_data_length / 8);
	
	//crc_bit
	unsigned char* crc_append = new unsigned char[24];
	byte2bit(crc_bytes, crc_append, 3);
	
	//Append crc
	for (int j = 0; j < 24; j++)
	{
		data[j + input_data_length] = crc_append[j];
	}
	delete(crc_bytes);
	delete(crc_append);
	delete(data_bytes);
}

//check crc
bool rx_check_crc(unsigned char* data, int input_datalength)
{
	//Check CRC
	unsigned char* databytes = new unsigned char[(input_datalength - 24) / 8];
	bit2byte(data, databytes, input_datalength-24);
	unsigned char* message_crc_bytes = new unsigned char[3];
 	bit2byte(data + input_datalength - 24, message_crc_bytes, 24);

	unsigned char* calculated_crc_bytes = new unsigned char[3];
	crc(databytes, calculated_crc_bytes, (input_datalength-24) / 8);

	bool crc_correct = (calculated_crc_bytes[0] == message_crc_bytes[0]) && (calculated_crc_bytes[1] == message_crc_bytes[1]) &&  (calculated_crc_bytes[2] == message_crc_bytes[2]);

	delete(databytes);
	delete(message_crc_bytes);
	delete(calculated_crc_bytes);
	return crc_correct;
}


//calculate crc
void crc(unsigned char* msg, unsigned char* output_bytes,int input_length)
{
	crc_t crc;

	crc = crc_init();
    	crc = crc_update(crc, msg, input_length);
    	crc = crc_finalize(crc);

	output_bytes[0] = (unsigned char)(crc  & 0x000000ff);
    	output_bytes[1] = (unsigned char)((crc & 0x0000ff00) >> 8);
   	output_bytes[2] = (unsigned char)((crc & 0x00ff0000) >> 16);
}

//Update the crc value with new data.
crc_t crc_update(crc_t crc, const unsigned char *data, size_t data_len)
{
	unsigned int tbl_idx;
	while (data_len--) {
		tbl_idx = ((crc >> 16) ^ *data) & 0xff;
		crc = (crc_table[tbl_idx] ^ (crc << 8)) & 0xffffff;
		data++;
	}
	return crc & 0xffffff;
}

//bit2byte
void bit2byte(unsigned char* input, unsigned char* output, int Data_length)  
{
	int output_length = Data_length/8;
	unsigned char current_byte;
	
	/* create the output vector */
	for (int i = 0; i < output_length; i++) {
		current_byte = input[i * 8];
		current_byte = current_byte + (input[i * 8 + 1] << 1);
		current_byte = current_byte + (input[i * 8 + 2] << 2);
		current_byte = current_byte + (input[i * 8 + 3] << 3);
		current_byte = current_byte + (input[i * 8 + 4] << 4);
		current_byte = current_byte + (input[i * 8 + 5] << 5);
		current_byte = current_byte + (input[i * 8 + 6] << 6);
		current_byte = current_byte + (input[i * 8 + 7] << 7);
		output[i] = current_byte;
	}
}

//byte2bit 
void byte2bit(unsigned char* input, unsigned char* output, int DataLength)
{
	int output_length = DataLength * 8;
	int output_index;

	/* create the output vector */
	for (int i = 0; i < DataLength; i++) {
		output_index = 8 * i;
		output[output_index] = input[i] & BIT_1;
		output[output_index + 1] = (input[i] & BIT_2) >> 1;
		output[output_index + 2] = (input[i] & BIT_3) >> 2;
		output[output_index + 3] = (input[i] & BIT_4) >> 3;
		output[output_index + 4] = (input[i] & BIT_5) >> 4;
		output[output_index + 5] = (input[i] & BIT_6) >> 5;
		output[output_index + 6] = (input[i] & BIT_7) >> 6;
		output[output_index + 7] = (input[i] & BIT_8) >> 7;
	}
}
