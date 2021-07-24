all: dense.o conv2d.o accum.o lstm.o gru.o pooling.o upres.o normalization.o neuron.o
.PHONY: all

dense.o: dense.h dense.cpp
	g++ -c -Wall -I ./ dense.cpp

conv2d.o: conv2d.h conv2d.cpp
	g++ -c -Wall -I ./ conv2d.cpp

accum.o: accum.h accum.cpp
	g++ -c -Wall -I ./ accum.cpp

lstm.o: lstm.h lstm.cpp
	g++ -c -Wall -I ./ lstm.cpp

gru.o: gru.h gru.cpp
	g++ -c -Wall -I ./ gru.cpp

pooling.o: pooling.h pooling.cpp
	g++ -c -Wall -I ./ pooling.cpp

upres.o: upres.h upres.cpp
	g++ -c -Wall -I ./ upres.cpp

normalization.o: normalization.h normalization.cpp
	g++ -c -Wall -I ./ normalization.cpp

neuron.o: neuron.h neuron.cpp dense.h dense.cpp conv2d.h conv2d.cpp accum.h accum.cpp lstm.h lstm.cpp gru.h gru.cpp pooling.h pooling.cpp upres.h upres.cpp normalization.h normalization.cpp
	g++ -c -Wall -I ./ dense.cpp
	g++ -c -Wall -I ./ conv2d.cpp
	g++ -c -Wall -I ./ accum.cpp
	g++ -c -Wall -I ./ lstm.cpp
	g++ -c -Wall -I ./ gru.cpp
	g++ -c -Wall -I ./ pooling.cpp
	g++ -c -Wall -I ./ upres.cpp
	g++ -c -Wall -I ./ normalization.cpp
	g++ -c -Wall -I ./ neuron.cpp
