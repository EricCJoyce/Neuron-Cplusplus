#ifndef __GRU_H
#define __GRU_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a GRU Layer as matrices and vectors:
  d = the length of a single input instance
      (that is, we may have an indefinitely long sequence of word-vectors, but each is an input instance of length 'd')
  h = the length of internal state vectors
  cache = the number of previous states to track and store

  input d-vec{x}       weights Wz              weights Wr              weights Wh
 [ x1 x2 x3 x4 ]        (h by d)                (h by d)                (h by d)
                 [ wz11 wz12 wz13 wz14 ] [ wr11 wr12 wr13 wr14 ] [ wh11 wh12 wh13 wh14 ]
                 [ wz21 wz22 wz23 wz24 ] [ wr21 wr22 wr23 wr24 ] [ wh21 wh22 wh23 wh24 ]
                 [ wz31 wz32 wz33 wz34 ] [ wr31 wr32 wr33 wr34 ] [ wh31 wh32 wh33 wh34 ]

                       weights Uz              weights Ur              weights Uh
                        (h by h)                (h by h)                (h by h)
                 [ uz11 uz12 uz13 ]      [ ur11 ur12 ur13 ]      [ uh11 uh12 uh13 ]
                 [ uz21 uz22 uz23 ]      [ ur21 ur22 ur23 ]      [ uh21 uh22 uh23 ]
                 [ uz31 uz32 uz33 ]      [ ur31 ur32 ur33 ]      [ uh31 uh32 uh33 ]

                     bias h-vec{bz}          bias h-vec{br}          bias h-vec{bh}
                 [ bz1 ]                 [ br1 ]                 [ bh1 ]
                 [ bz2 ]                 [ br2 ]                 [ bh2 ]
                 [ bz3 ]                 [ br3 ]                 [ bh3 ]

         H state cache (times 1, 2, 3, 4 = columns 0, 1, 2, 3)
        (h by cache)
 [ H11 H12 H13 H14 ]
 [ H21 H22 H23 H24 ]
 [ H31 H32 H33 H34 ]

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

#include <iostream>
#include <Eigen/Dense>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */

/*
#define __GRU_DEBUG 1
*/

using Eigen::MatrixXd;
using namespace std;

/**************************************************************************************************
 Typedefs  */


/**************************************************************************************************
 GRU  */
class GRU
  {
    public:
      GRU(unsigned int, unsigned int, unsigned int);                //  Constructor(s)
      ~GRU();                                                       //  Destructor

      void setWz(double*);                                          //  Set entirety of Wz weight matrix
      void setWr(double*);                                          //  Set entirety of Wr weight matrix
      void setWh(double*);                                          //  Set entirety of Wh weight matrix

      void setWz_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Wz weight matrix
      void setWr_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Wr weight matrix
      void setWh_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Wh weight matrix

      void setUz(double*);                                          //  Set entirety of Uz weight matrix
      void setUr(double*);                                          //  Set entirety of Ur weight matrix
      void setUh(double*);                                          //  Set entirety of Uh weight matrix

      void setUz_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Uz weight matrix
      void setUr_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Ur weight matrix
      void setUh_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Uh weight matrix

      void setbz(double*);                                          //  Set entirety of bz bias vector
      void setbr(double*);                                          //  Set entirety of br bias vector
      void setbh(double*);                                          //  Set entirety of bh bias vector

      void setbz_i(double, unsigned int);                           //  Set i-th element of bz bias vector
      void setbr_i(double, unsigned int);                           //  Set i-th element of br bias vector
      void setbh_i(double, unsigned int);                           //  Set i-th element of bh bias vector

      void setName(char*);
      char* name() const;
      void print() const;
      unsigned int outputLen() const;
      unsigned int run(double*);
      void reset();

    private:
      unsigned int d;                                               //  Dimensionality of input vector
      unsigned int h;                                               //  Dimensionality of hidden state vector
      unsigned int cache;                                           //  The number of states to keep in memory:
                                                                    //  when 't' exceeds this, shift out.
      unsigned int t;                                               //  The time step
                                                                    //  W matrices are (h by d)
      MatrixXf Wz;
      MatrixXf Wr;
      MatrixXf Wh;
                                                                    //  U matrices are (h by h)
      MatrixXf Uz;
      MatrixXf Ur;
      MatrixXf Uh;
                                                                    //  Bias vectors are length h
      VectorXf bz;
      VectorXf br;
      VectorXf bh;

      MatrixXf H;                                                   //  Hidden state cache matrix (h by cache)
      char name[LAYER_NAME_LEN];
  };

#endif