#ifndef __UPRES_H
#define __UPRES_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 An upres layer serves to prepare input for (transposed) convolution.
  s = stride
  p = padding

    input mat{X}         output for s = 1, p = 0        output for s = 1, p = 1
 [ x11 x12 x13 x14 ]    [ x11 0 x12 0 x13 0 x14 ]    [ 0  0  0  0  0  0  0  0  0 ]
 [ x21 x22 x23 x24 ]    [  0  0  0  0  0  0  0  ]    [ 0 x11 0 x12 0 x13 0 x14 0 ]
 [ x31 x32 x33 x34 ]    [ x21 0 x22 0 x23 0 x24 ]    [ 0  0  0  0  0  0  0  0  0 ]
 [ x41 x42 x43 x44 ]    [  0  0  0  0  0  0  0  ]    [ 0 x21 0 x22 0 x23 0 x24 0 ]
 [ x51 x52 x53 x54 ]    [ x31 0 x32 0 x33 0 x34 ]    [ 0  0  0  0  0  0  0  0  0 ]
                        [  0  0  0  0  0  0  0  ]    [ 0 x31 0 x32 0 x33 0 x34 0 ]
                        [ x41 0 x42 0 x43 0 x44 ]    [ 0  0  0  0  0  0  0  0  0 ]
                        [  0  0  0  0  0  0  0  ]    [ 0 x41 0 x42 0 x43 0 x44 0 ]
                        [ x51 0 x52 0 x53 0 x54 ]    [ 0  0  0  0  0  0  0  0  0 ]
                                                     [ 0 x51 0 x52 0 x53 0 x54 0 ]
                                                     [ 0  0  0  0  0  0  0  0  0 ]

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

#include <iostream>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILL_ZERO    0                                              /* Fill strides or pad using zeroes */
#define FILL_SAME    1                                              /* Fill strides or pad using duplicates of the nearest value */
#define FILL_INTERP  2                                              /* Fill strides or pad using bilinear interpolation */

#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */

/*
#define __UPRES_DEBUG 1
*/

using namespace std;

/**************************************************************************************************
 Typedefs  */

typedef struct UpresParamsType
  {
    unsigned int stride_h;                                          //  Horizontal Stride: number of columns to put between input columns
    unsigned int stride_v;                                          //  Vertical Stride: number of rows to put between input rows
    unsigned int padding_h;                                         //  Horizontal Padding: depth of pixels appended to the source border, left and right
    unsigned int padding_v;                                         //  Vertical Padding: depth of pixels appended to the source border, top and bottom

    unsigned char sMethod;                                          //  In {FILL_ZERO, FILL_SAME, FILL_INTERP}
    unsigned char pMethod;
  } UpresParams;

/**************************************************************************************************
 Upres  */
class Upres
  {
    public:
      Upres(unsigned int, unsigned int);                            //  Constructor(s)
      ~Upres();                                                     //  Destructor

      unsigned int addParams(unsigned int, unsigned int);
      void setParamsHorzStride(unsigned int, unsigned int);
      void setParamsVertStride(unsigned int, unsigned int);
      void setParamsHorzPad(unsigned int, unsigned int);
      void setParamsVertPad(unsigned int, unsigned int);
      void setParamsStrideMethod(unsigned char, unsigned int);
      void setParamsPaddingMethod(unsigned char, unsigned int);

      void setName(char*);
      char* name() const;
      void print() const;
      unsigned int outputLen() const;
      unsigned int run(double*);

    private:
      unsigned int inputW;                                          //  Dimensions of the input
      unsigned int inputH;                                    
      UpresParams* params;                                          //  Array of Up-resolution parameters structures
      unsigned int n;                                               //  Number of up-ressings in this layer

      char name[LAYER_NAME_LEN];
      unsigned int outlen;                                          //  Length of the output buffer
      double* out;
  };

#endif