#ifndef __DENSE_CPP
#define __DENSE_CPP

#include "dense.h"

/**************************************************************************************************
 Constructor(s)/Destructor  */

/*  */
Dense::Dense(unsigned int inputs, unsigned int nodes)
  {
    unsigned int x, y;

    W = new MatrixXf(inputs + 1, nodes);
    M = new MatrixXf(inputs + 1, nodes);
    out = new VectorXf(nodes);
    if((f = (unsigned char*)malloc(nodes * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate Dense layer's internal output array\n";
        exit(1);
      }
    if((alpha = (double*)malloc(nodes * sizeof(double))) == NULL)
      {
        cout << "ERROR: Unable to allocate Dense layer's function-parameter array\n";
        exit(1);
      }

    for(y = 0; y < (inputs + 1); y++)                               //  Generate random numbers in [ -1.0, 1.0 ]
      {
        for(x = 0; x < nodes; x++)
          {
            W(y, x) = -1.0 + ((double)rand() / ((double)RAND_MAX * 0.5));
            M(y, x) = 1.0;                                          //  All are UNmasked
          }
      }

    for(x = 0; x < nodes; x++)                                      //  Default all to ReLU with parameter = 1.0
      {
        f[x] = RELU;
        alpha[x] = 1.0;
      }

    for(x = 0; x < LAYER_NAME_LEN; x++)                             //  Blank out layer name
      layerName[x] = '\0';
  }

Dense::~Dense()
  {
    delete W;
    delete M;
    delete out;
    free(f);
    free(alpha);
  }

/**************************************************************************************************
 Weight matrix  */

/*  */
void Dense::setW(double*)
  {
  }

/*  */
void Dense::setW_i(double*, unsigned int)
  {
  }

/*  */
void Dense::setW_ij(double, unsigned int, unsigned int)
  {
  }

/**************************************************************************************************
 Mask matrix  */

/*  */
void Dense::setM(bool*)
  {
  }

/*  */
void Dense::setM_i(bool*, unsigned int)
  {
  }

/*  */
void Dense::setM_ij(bool, unsigned int, unsigned int)
  {
  }

/**************************************************************************************************
 Other setters  */

/*  */
void Dense::setF_i(unsigned char, unsigned int)
  {
  }

/*  */
void Dense::setA_i(double, unsigned int)
  {
  }

/*  */
void Dense::setName(char*)
  {
  }

/**************************************************************************************************
 Display  */

/*  */
void Dense::print() const
  {
  }

/*  */
unsigned int Dense::outputLen() const
  {
  }

/**************************************************************************************************
 Run layer  */

/*  */
unsigned int Dense::run(double*)
  {
  }

#endif