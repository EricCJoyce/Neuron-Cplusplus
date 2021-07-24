#ifndef __ACCUM_H
#define __ACCUM_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

#include <iostream>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */

/*
#define __ACCUM_DEBUG 1
*/

using namespace std;

/**************************************************************************************************
 Typedefs  */


/**************************************************************************************************
 Accum  */
class Accum
  {
    public:
      Accum(unsigned int);                                          //  Constructor(s)
      ~Accum();                                                     //  Destructor

      void setName(char*);
      char* name() const;
      void print() const;
      unsigned int outputLen() const;
      unsigned int run(double*);

    private:
      unsigned int inputs;                                          //  Number of inputs--ACCUMULATORS GET NO bias-1
      char name[LAYER_NAME_LEN];
      double* out;
  };