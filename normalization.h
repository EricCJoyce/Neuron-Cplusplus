#ifndef __NORMAL_H
#define __NORMAL_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 A normalizing layer applies the four learned parameters to its input.
  m = learned mean
  s = learned standard deviation
  g = learned coefficient
  b = learned constant

 input vec{x}    output vec{y}
   [ x1 ]     [ g*((x1 - m)/s)+b ]
   [ x2 ]     [ g*((x2 - m)/s)+b ]
   [ x3 ]     [ g*((x3 - m)/s)+b ]
   [ x4 ]     [ g*((x4 - m)/s)+b ]
   [ x5 ]     [ g*((x5 - m)/s)+b ]

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

#include <iostream>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */

/*
#define __NORMALIZE_DEBUG 1
*/

using namespace std;

/**************************************************************************************************
 Typedefs  */


/**************************************************************************************************
 Normalization  */
class Normalization
  {
    public:
      Normalization(unsigned int);                                  //  Constructor(s)
      ~Normalization();                                             //  Destructor

      void setM(double);
      void setS(double);
      void setG(double);
      void setB(double);

      void setName(char*);
      char* name() const;
      void print() const;
      unsigned int outputLen() const;
      unsigned int run(double*);

    private:
      unsigned int inputs;                                          //  Number of inputs--ACCUMULATORS GET NO bias-1
      double m;                                                     //  Mu: the mean learned during training
      double s;                                                     //  Sigma: the standard deviation learned during training
      double g;                                                     //  The factor learned during training
      double b;                                                     //  The constant learned during training
      char name[LAYER_NAME_LEN];
      double* out;
  };