#pragma once
#include "Utility.h"

struct Neuron
{
	std::vector<float> input;
	std::vector<float>output;
	std::vector<float> weights;
	float bias;
	int inputSize, outputSize;

	Neuron(int inputSize, int outputSize, std::vector<float> input) : inputSize(inputSize), outputSize(outputSize), input(input)
	{
		bias = 0.f;
		for (int i = 0; i < this->inputSize; i++)
		{
			this->weights.push_back(rx::Utility::randFloat(10, 20) / static_cast<float>(100.f));
		}
	}
};

class InputLayer
{
private:
	std::vector<float> input;
	std::vector<Neuron*> neurons;
	std::vector<float> output;

	int inputSize, outputSize;
public:
	InputLayer(int inputSize, int outputSize, std::vector<float> input) : outputSize(outputSize), inputSize(inputSize), input(input)
	{
		for (int i = 0; i < inputSize; i++)
		{
			this->neurons.push_back(new Neuron(this->inputSize, this->outputSize, this->input));
		}
	}
	void forward()
	{
		for (const auto& e : this->neurons)
		{
			output.push_back(rx::Utility::ReLU(rx::Utility::dot(this->input, e->weights) + e->bias));
		}
	}
	std::vector<float>& getOutput()
	{
		return this->output;
	}

	void setInput(std::vector<float> input)
	{
		this->input = input;
	}
};

class OutputLayer
{
private:
	std::vector<float> input;
	Neuron* neuron;
	float output;
	int inputSize, outputSize;
	int batchSize;
	std::vector<float> tempLoss;
	int count = 0;
	int totalBatches = 0;
	std::vector<float> y;
	float cost;
public:

	OutputLayer(int inputSize, int outputSize, std::vector<float> input, std::vector<float> y, int batchSize) :
		outputSize(outputSize), inputSize(inputSize), input(input), batchSize(batchSize), y(y)
	{
		neuron = new Neuron(inputSize, outputSize, input);
	}
	void forward()
	{
		
		output = (rx::Utility::Sigmoid(rx::Utility::dot(this->input, neuron->weights) + neuron->bias));
		if (count < batchSize)
		{
			this->tempLoss.push_back(rx::Utility::loss(output, y[count + totalBatches]));
			totalBatches += this->batchSize;
			this->count++;
		}
		else
		{
			this->count = 0;
			this->cost = rx::Utility::cost(tempLoss);
			this->tempLoss.clear();
		}
	}
	float getOutput()
	{
		return this->output;
	}
	float getCost()
	{
		return this->cost;
	}

	void setInput(std::vector<float> input)
	{
		this->input = input;
	}

};

class HiddenLayer
{
private:
	std::vector<float> input;
	std::vector<Neuron*> neurons;
	std::vector<float> output;

	int inputSize, outputSize;
public:
	HiddenLayer(int inputSize, int outputSize, std::vector<float> input) : outputSize(outputSize), inputSize(inputSize), input(input)
	{
		for (int i = 0; i < inputSize; i++)
		{
			this->neurons.push_back(new Neuron(this->inputSize, this->outputSize, this->input));
		}
	}
	void forward()
	{
		for (const auto& e : this->neurons)
		{
			output.push_back(rx::Utility::ReLU(rx::Utility::dot(this->input, e->weights) + e->bias));
		}
	}
	std::vector<float>& getOutput()
	{
		return this->output;
	}
};