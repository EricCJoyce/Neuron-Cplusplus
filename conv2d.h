#ifndef __CONV2D_H
#define __CONV2D_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a Convolutional Layer as an array of one or more 2D filters:

  input mat{X} w, h       filter1      activation function vector f
 [ x11 x12 x13 x14 ]    [ w11 w12 ]   [ func1 func2 ]
 [ x21 x22 x23 x24 ]    [ w21 w22 ]
 [ x31 x32 x33 x34 ]    [ bias ]       auxiliary vector alpha
 [ x41 x42 x43 x44 ]                  [ param1 param2 ]
 [ x51 x52 x53 x54 ]      filter2
                      [ w11 w12 w13 ]
                      [ w21 w22 w23 ]
                      [ w31 w32 w33 ]
                      [ bias ]

 Filters needn't be arranged from smallest to largest; this is just for illustration.

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

#include <iostream>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RELU                 0                                      /* [ 0.0, inf) */
#define LEAKY_RELU           1                                      /* (-inf, inf) */
#define SIGMOID              2                                      /* ( 0.0, 1.0) */
#define HYPERBOLIC_TANGENT   3                                      /* [-1.0, 1.0] */
#define SOFTMAX              4                                      /* [ 0.0, 1.0] */
#define SYMMETRICAL_SIGMOID  5                                      /* (-1.0, 1.0) */
#define THRESHOLD            6                                      /* { 0.0, 1.0} */
#define LINEAR               7                                      /* (-inf, inf) */

#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */

/*
#define __CONV2D_DEBUG 1
*/

using namespace std;

/**************************************************************************************************
 Typedefs  */

typedef struct Filter2DType
  {
    unsigned int w;                                                 //  Width of the filter
    unsigned int h;                                                 //  Height of the filter

    unsigned int stride_h;                                          //  Stride by which we move the filter left to right
    unsigned int stride_v;                                          //  Stride by which we move the filter top to bottom

    unsigned char f;                                                //  Function flag, in {RELU, LEAKY_RELU, ..., THRESHOLD, LINEAR}
    double alpha;                                                   //  Function parameter (not always applicable)

    double* W;                                                      //  Array of (w * h) weights, arranged row-major, +1 for the bias
  } Filter2D;

/**************************************************************************************************
 Conv2D  */
class Conv2D
  {
    public:
      Conv2D(unsigned int, unsigned int);                           //  Constructor(s)
      ~Conv2D();                                                    //  Destructor

      unsigned int addFilter(unsigned int, unsigned int);           //  Add a filter to the layer
      void setW_i(double*, unsigned int);                           //  Set entirety of i-th filter; w is length width * height + 1
      void setW_ij(double, unsigned int, unsigned int);             //  Set the j-th weight of the i-th filter
      void setHorzStride_i(unsigned int, unsigned int);             //  Set the horizontal stride of the i-the filter
      void setVertStride_i(unsigned int, unsigned int);             //  Set the vertical stride of the i-the filter
      void setF_i(unsigned char, unsigned int);                     //  Set activation function of i-th filter
      void setA_i(double, unsigned int);                            //  Set activation function parameter of i-th filter
      void setName(char*);
      char* name() const;
      void print() const;
      unsigned int outputLen() const;
      unsigned int run(double*);

    private:
      unsigned int inputW;                                          //  Dimensions of the input
      unsigned int inputH;                                    
      unsigned int n;                                               //  Number of processing units in this layer =
                                                                    //  number of filters in this layer
      Filter2D* filters;                                            //  Array of 2D filter structs

      char name[LAYER_NAME_LEN];
      unsigned int outlen;                                          //  Length of the output buffer
      double* out;
  };

#endif