#ifndef __NEURON_H
#define __NEURON_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

#include <iostream>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "accum.h"                                                  /* Include Accumulator Layer library */
#include "conv2d.h"                                                 /* Include 2D-Convolutional Layer library */
#include "dense.h"                                                  /* Include Dense Layer library */
#include "gru.h"                                                    /* Include GRU Layer library */
#include "lstm.h"                                                   /* Include LSTM Layer library */
#include "normalization.h"                                          /* Include Normalization Layer library */
#include "pooling.h"                                                /* Include Pooling Layer library */
#include "upres.h"                                                  /* Include Up-Res (a.k.a. Transpose Convolution) Layer library */

#define INPUT_ARRAY   0                                             /* Flag refers to network input */
#define DENSE_ARRAY   1                                             /* Flag refers to 'denselayers' */
#define CONV2D_ARRAY  2                                             /* Flag refers to 'convlayers' */
#define ACCUM_ARRAY   3                                             /* Flag refers to 'accumlayers' */
#define LSTM_ARRAY    4                                             /* Flag refers to 'lstmlayers' */
#define GRU_ARRAY     5                                             /* Flag refers to 'grulayers' */
#define POOL_ARRAY    6                                             /* Flag refers to 'poollayers' */
#define UPRES_ARRAY   7                                             /* Flag refers to 'upreslayers' */
#define NORMAL_ARRAY  8                                             /* Flag refers to 'normallayers' */

#define VARSTR_LEN      16                                          /* Length of a Variable key string */
#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */
#define COMMSTR_LEN     64                                          /* Length of a Network Comment string */

/*
#define __NEURON_DEBUG 1
*/

using namespace std;

/**************************************************************************************************
 Typedefs  */

typedef struct VariableType
  {
    char key[VARSTR_LEN];                                           //  String for variable key/symbol
    double value;                                                   //  Variable's value
  } Variable;

typedef struct NodeType                                             //  Really just used in connectivity tests
  {
    unsigned char type;                                             //  Which network array to look in
    unsigned int index;                                             //  Index into that array
  } Node;

typedef struct EdgeType
  {
    unsigned char srcType;                                          //  Indicates in which array to find the source
    unsigned int srcIndex;                                          //  Index into that array

    unsigned int selectorStart;                                     //  From (and including) this array element...
    unsigned int selectorEnd;                                       //  ...to (but excluding) this array element.

    unsigned char dstType;                                          //  Indicates in which array to find the destination
    unsigned int dstIndex;                                          //  Index into that array
  } Edge;

/**************************************************************************************************
 NeuralNet  */
class NeuralNet
  {
    public:
      NeuralNet(unsigned int);                                      //  Constructor(s)
      ~NeuralNet();                                                 //  Destructor

      unsigned int run(double*, double**);
      bool linkLayers(unsigned char, unsigned int, unsigned int, unsigned int, unsigned char, unsigned int);
      bool load(char*);
      bool write(char*);
      void sortEdges();
      unsigned int nameIndex(char*);
      unsigned char nameType(char*);
      void printEdgeList();
      void print();
      void printLayerName(unsigned char, unsigned int);

      unsigned int addDense(unsigned int, unsigned int);
      unsigned int addConv2D(unsigned int, unsigned int);
      unsigned int addAccum(unsigned int);
      unsigned int addLSTM(unsigned int, unsigned int, unsigned int);
      unsigned int addGRU(unsigned int, unsigned int, unsigned int);
      unsigned int addPool(unsigned int, unsigned int);
      unsigned int addUpres(unsigned int, unsigned int);
      unsigned int addNormal(unsigned int);

    private:
      unsigned int inputs;                                          //  Number of Network inputs

      Edge* edgelist;                                               //  Edge list
      unsigned int len;                                             //  Length of edge list

      DenseLayer* denselayers;                                      //  Array of Dense Layers
      unsigned int denseLen;                                        //  Length of that array

      Conv2DLayer* convlayers;                                      //  Array of Conv2D Layers
      unsigned int convLen;                                         //  Length of that array

      AccumLayer* accumlayers;                                      //  Array of Accum Layers
      unsigned int accumLen;                                        //  Length of that array

      LSTMLayer* lstmlayers;                                        //  Array of LSTM Layers
      unsigned int lstmLen;                                         //  Length of that array

      GRULayer* grulayers;                                          //  Array of GRU Layers
      unsigned int gruLen;                                          //  Length of that array

      Pool2DLayer* poollayers;                                      //  Array of Pooling Layers
      unsigned int poolLen;                                         //  Length of that array

      UpresLayer* upreslayers;                                      //  Array of Upres Layers
      unsigned int upresLen;                                        //  Length of that array

      NormalLayer* normlayers;                                      //  Array of Normal Layers
      unsigned int normalLen;                                       //  Length of that array

      Variable* variables;                                          //  Array of Network Variables
      unsigned char vars;                                           //  Length of that array

      unsigned int gen;                                             //  Network generation/epoch
      double fit;                                                   //  Network fitness
      char comment[COMMSTR_LEN];                                    //  Network comment
  };

#endif  