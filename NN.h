#pragma once
#include "Layers.h"
#include "DataSet.h"



class NN
{
private:
	InputLayer* input_layer;
	HiddenLayer* hidden_layer;
	OutputLayer* output_layer;
	rx::DataSet& dataset;
	rx::SET& set;
public:
	NN(rx::DataSet& dataset);

	void addInputLayer(int inputSize, int outputSize);
	void addHiddenLayer(int inputSize, int outputSize);
	void addOutputLayer(int inputSize, int outputSize);
};

