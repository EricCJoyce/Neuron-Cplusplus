#ifndef __DENSE_H
#define __DENSE_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a Dense Layer as two matrices and two vectors:

    input vec{x}         weights W             masks M
 [ x1 x2 x3 x4 1 ]  [ w11 w12 w13 w14 ]  [ m11 m12 m13 m14 ]
                    [ w21 w22 w23 w24 ]  [ m21 m22 m23 m24 ]
                    [ w31 w32 w33 w34 ]  [ m31 m32 m33 m34 ]
                    [ w41 w42 w43 w44 ]  [ m41 m42 m43 m44 ]
                    [ w51 w52 w53 w54 ]  [  1   1   1   1  ]

                    activation function
                         vector f
               [ func1 func2 func3 func4 ]

                     auxiliary vector
                          alpha
               [ param1 param2 param3 param4 ]

 Broadcast W and M = W'
 vec{x} dot W' = x'
 vec{output} is func[i](x'[i], param[i]) for each i

 Not all activation functions need a parameter. It's just a nice feature we like to offer.

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

#include <iostream>
#include <Eigen/Dense>
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
#define __DENSE_DEBUG 1
*/

using Eigen::MatrixXd;
using namespace std;

/**************************************************************************************************
 Typedefs  */


/**************************************************************************************************
 Dense  */
class Dense
  {
    public:
      Dense(unsigned int, unsigned int);                            //  Constructor(s)
      ~Dense();                                                     //  Destructor

      void setW(double*);                                           //  Set entirety of layer's weight matrix
      void setW_i(double*, unsigned int);                           //  Set entirety of weights for i-th column/neuron/unit
      void setW_ij(double, unsigned int, unsigned int);             //  Set element [i, j] of layer's weight matrix
      void setM(bool*);                                             //  Set entirety of layer's mask matrix
      void setM_i(bool*, unsigned int);                             //  Set entirety of masks for i-th column/neuron/unit
      void setM_ij(bool, unsigned int, unsigned int);               //  Set element [i, j] of layer's mask matrix
      void setF_i(unsigned char, unsigned int);                     //  Set activation function of i-th neuron/unit
      void setA_i(double, unsigned int);                            //  Set activation function auxiliary parameter of i-th neuron/unit
      void setName(char*);
      char* name() const;
      void print() const;
      unsigned int outputLen() const;
      unsigned int run(double*);

    private:
      unsigned int inputs;                                          //  Number of inputs--NOT COUNTING the added bias-1
      unsigned int nodes;                                           //  Number of processing units in this layer
      MatrixXf W;                                                   //  ((i + 1) x n) matrix
      MatrixXf M;                                                   //  ((i + 1) x n) matrix, all either 0.0 or 1.0
      unsigned char* f;                                             //  n-array
      double* alpha;                                                //  n-array
      char layerName[LAYER_NAME_LEN];
      VectorXf out;                                                 //  (n x 1) matrix
  };

#endif