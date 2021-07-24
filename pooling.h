#ifndef __POOLING_H
#define __POOLING_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a Pooling Layer as 2D input dimensions and an array of 2D pools.
  inputW = width of the input
  inputH = height of the input

 Each pool has a 2D shape, two dimensions for stride, and a function/type:
  stride_h = horizontal stride of the pool
  stride_v = vertical stride of the pool
  f = {MAX_POOL, AVG_POOL, MIN_POOL, MEDIAN_POOL}

    input mat{X}          pool     output for s = (1, 1)     output for s = (2, 2)
 [ x11 x12 x13 x14 ]    [ . . ]   [ y11  y12  y13 ]         [ y11  y12 ]
 [ x21 x22 x23 x24 ]    [ . . ]   [ y21  y22  y23 ]         [ y21  y22 ]
 [ x31 x32 x33 x34 ]              [ y31  y32  y33 ]
 [ x41 x42 x43 x44 ]              [ y41  y42  y43 ]
 [ x51 x52 x53 x54 ]

 Pools needn't be arranged from smallest to largest or in any order.

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

#include <iostream>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_POOL     0
#define MIN_POOL     1
#define AVG_POOL     2
#define MEDIAN_POOL  3

#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */

/*
#define __POOLING_DEBUG 1
*/

using namespace std;

/**************************************************************************************************
 Typedefs  */

typedef struct Pool2DType
  {
    unsigned int w;                                                 //  Width of the pool
    unsigned int h;                                                 //  Height of the pool
    unsigned int stride_h;                                          //  Stride by which we move the pool left to right
    unsigned int stride_v;                                          //  Stride by which we move the pool top to bottom
    unsigned char f;                                                //  In {MAX_POOL, MIN_POOL, AVG_POOL, MEDIAN_POOL}
  } Pool2D;

/**************************************************************************************************
 Pooling  */
class Pooling
  {
    public:
      Pooling(unsigned int, unsigned int);                          //  Constructor(s)
      ~Pooling();                                                   //  Destructor

      unsigned int addPool(unsigned int, unsigned int);
      void setPoolWidth(unsigned int, unsigned int);
      void setPoolHeight(unsigned int, unsigned int);
      void setPoolHorzStride(unsigned int, unsigned int);
      void setPoolVertStride(unsigned int, unsigned int);
      void setPoolFunc(unsigned char, unsigned int);
      void setName(char*);
      char* name() const;
      void print() const;
      unsigned int outputLen() const;
      unsigned int run(double*);

    private:
      unsigned int inputW;                                          //  Dimensions of the input
      unsigned int inputH;
      Pool2D* pools;                                                //  Array of Pool2Ds
      unsigned int n;                                               //  Length of that array

      double* out;
      unsigned int outlen;                                          //  Length of the output buffer

      char name[LAYER_NAME_LEN];

      void pooling_quicksort(bool, double**, unsigned int, unsigned int);
      unsigned int pooling_partition(bool, double**, unsigned int, unsigned int);
  };

#endif