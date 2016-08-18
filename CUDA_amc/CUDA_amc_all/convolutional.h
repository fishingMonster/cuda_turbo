/* File convolutional.h
   
   Description: General functions used to implement convolutional encoding. 
   MEXed to LTE_tx_convolutional_encoder

   Copyright (C) 2006-2008, Matthew C. Valenti

   Last updated on May 22, 2008

   The functions in this file are part of the Iterative Solutions 
   Coded Modulation Library. The Iterative Solutions Coded Modulation 
   Library is free software; you can redistribute it and/or modify it 
   under the terms of the GNU Lesser General Public License as published 
   by the Free Software Foundation; either version 2.1 of the License, 
   or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

*/
#define MAXLOG 1e7  /* Define infinity */

/* function itob()

  Description: Converts an integer symbol into a vector of bits

	Output parameters:
		binvec_p: The binary vector
		
    Input parameters:
	    symbol:  The integer-valued symbol
		length:  The length of the binary vector
	
  This function is used by conv_encode()  */

void itob(
						int	binvec_p[],
						int symbol,
						int length )
{
	int counter;

	/* Go through each bit in the vector */
	for (counter=0;counter<length;counter++) {
		binvec_p[length-counter-1] = (symbol&1);
		symbol = symbol>>1;
	}

	return;
}


/* perform convolutional encoding */
static void conv_encode(
	     int		output_p[],
	     int		input[],
		 int		out0[], 
		 int		state0[], 
		 int		out1[], 
		 int		state1[],
         int		tail[],	 
         int        KK,
         int        LL,
		 int        nn )
{
  int i, j, outsym;
  int *bin_vec = new int[nn];
  int state = 0;

  /* Negative value in "tail" is a flag that this is 
  a tail-biting NSC code.  Determine initial state */

  if ( tail[0] < 0 ) {
	  for (i=LL-KK+1;i<LL;i++) {  
		  if (input[i]) {			  
			  /* Determine next state */
			  state = state1[state];
		  } else {	  
			  /* Determine next state */
			  state = state0[state];
		  }
	  }
  }

  
  /* encode data bits one bit at a time */
  for (i=0;i<LL;i++) {
   
	  if (input[i]) {
		  /* Input is a one */
		  outsym = out1[state];  /* The output symbol */
		  
		  /* Determine next state */
		  state = state1[state];
	  } else {
		  /* Input is a zero */
		  outsym = out0[state];  /* The output symbol */
		  
		  /* Determine next state */
		  state = state0[state];
	  }
	  /* Convert symbol to a binary vector	*/
	  itob( bin_vec, outsym, nn );
		
	  /* Assign to output */
	  for (j=0;j<nn;j++)
		  output_p[nn*i+j] = bin_vec[j];
  }

  /* encode tail if needed */
  if (tail[0] >= 0) {
	  for (i=LL;i<LL+KK-1;i++) {
        
		  if (tail[state]) {
			  /* Input is a one */
			  outsym = out1[state];  /* The output symbol */
			  
			  /* Determine next state */
			  state = state1[state];
		  } else {
			  /* Input is a zero */
			  outsym = out0[state];  /* The output symbol */
			  
			  /* Determine next state */
			  state = state0[state];
		  }
		  
		  /* Convert symbol to a binary vector	*/
		  itob( bin_vec, outsym, nn );
		  
		  /* Assign to output */
		  for (j=0;j<nn;j++)
			  output_p[nn*i+j] = bin_vec[j];
	  }
  }

  delete[] bin_vec;


  return;
}



