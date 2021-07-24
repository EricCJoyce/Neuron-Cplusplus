#ifndef __LSTM_H
#define __LSTM_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model an LSTM Layer as matrices and vectors:
  d = the length of a single input instance
      (that is, we may have an indefinitely long sequence of word-vectors, but each is an input instance of length 'd')
  h = the length of internal state vectors
  cache = the number of previous states to track and store

  input d-vec{x}       weights Wi              weights Wo              weights Wf              weights Wc
 [ x1 x2 x3 x4 ]        (h by d)                (h by d)                (h by d)                (h by d)
                 [ wi11 wi12 wi13 wi14 ] [ wo11 wo12 wo13 wo14 ] [ wf11 wf12 wf13 wf14 ] [ wc11 wc12 wc13 wc14 ]
                 [ wi21 wi22 wi23 wi24 ] [ wo21 wo22 wo23 wo24 ] [ wf21 wf22 wf23 wf24 ] [ wc21 wc22 wc23 wc24 ]
                 [ wi31 wi32 wi33 wi34 ] [ wo31 wo32 wo33 wo34 ] [ wf31 wf32 wf33 wf34 ] [ wc31 wc32 wc33 wc34 ]

                       weights Ui              weights Uo              weights Uf              weights Uc
                        (h by h)                (h by h)                (h by h)                (h by h)
                 [ ui11 ui12 ui13 ]      [ uo11 uo12 uo13 ]      [ uf11 uf12 uf13 ]      [ uc11 uc12 uc13 ]
                 [ ui21 ui22 ui23 ]      [ uo21 uo22 uo23 ]      [ uf21 uf22 uf23 ]      [ uc21 uc22 uc23 ]
                 [ ui31 ui32 ui33 ]      [ uo31 uo32 uo33 ]      [ uf31 uf32 uf33 ]      [ uc31 uc32 uc33 ]

                     bias h-vec{bi}          bias h-vec{bo}          bias h-vec{bf}          bias h-vec{bc}
                 [ bi1 ]                 [ bo1 ]                 [ bf1 ]                 [ bc1 ]
                 [ bi2 ]                 [ bo2 ]                 [ bf2 ]                 [ bc2 ]
                 [ bi3 ]                 [ bo3 ]                 [ bf3 ]                 [ bc3 ]

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
#define __LSTM_DEBUG 1
*/

using Eigen::MatrixXd;
using namespace std;

/**************************************************************************************************
 Typedefs  */


/**************************************************************************************************
 LSTM  */
class LSTM
  {
    public:
      LSTM(unsigned int, unsigned int, unsigned int);               //  Constructor(s)
      ~LSTM();                                                      //  Destructor

      void setWi(double*);                                          //  Set entirety of Wi weight matrix
      void setWo(double*);                                          //  Set entirety of Wo weight matrix
      void setWf(double*);                                          //  Set entirety of Wf weight matrix
      void setWc(double*);                                          //  Set entirety of Wc weight matrix
      void setWi_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Wi weight matrix
      void setWo_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Wo weight matrix
      void setWf_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Wf weight matrix
      void setWc_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Wc weight matrix
      void setUi(double*);                                          //  Set entirety of Ui weight matrix
      void setUo(double*);                                          //  Set entirety of Uo weight matrix
      void setUf(double*);                                          //  Set entirety of Uf weight matrix
      void setUc(double*);                                          //  Set entirety of Uc weight matrix
      void setUi_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Ui weight matrix
      void setUo_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Uo weight matrix
      void setUf_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Uf weight matrix
      void setUc_ij(double, unsigned int, unsigned int);            //  Set element [i, j] of Uc weight matrix
      void setbi(double*);                                          //  Set entirety of bi bias vector
      void setbo(double*);                                          //  Set entirety of bo bias vector
      void setbf(double*);                                          //  Set entirety of bf bias vector
      void setbc(double*);                                          //  Set entirety of bc bias vector
      void setbi_i(double, unsigned int);                           //  Set i-th element of bi bias vector
      void setbo_i(double, unsigned int);                           //  Set i-th element of bo bias vector
      void setbf_i(double, unsigned int);                           //  Set i-th element of bf bias vector
      void setbc_i(double, unsigned int);                           //  Set i-th element of bc bias vector
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
      MatrixXf Wi;                                                  //  Input gate weights
      MatrixXf Wo;                                                  //  Output gate weights
      MatrixXf Wf;                                                  //  Forget gate weights
      MatrixXf Wc;                                                  //  Memory cell weights
                                                                    //  U matrices are (h by h)
      MatrixXf Ui;                                                  //  Recurrent connection input gate weights
      MatrixXf Uo;                                                  //  Recurrent connection output gate weights
      MatrixXf Uf;                                                  //  Recurrent connection forget gate weights
      MatrixXf Uc;                                                  //  Recurrent connection memory cell weights
                                                                    //  Bias vectors are length h
      VectorXf bi;                                                  //  Input gate bias
      VectorXf bo;                                                  //  Output gate bias
      VectorXf bf;                                                  //  Forget gate bias
      VectorXf bc;                                                  //  Memory cell bias

      VectorXf c;                                                   //  Cell state vector, length h
      MatrixXf H;                                                   //  Hidden state cache matrix (h by cache)
      char name[LAYER_NAME_LEN];
  };

#endif