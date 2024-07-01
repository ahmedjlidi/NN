#pragma once
#include "Tensor.h"
#include "Utility.h"

#define BATCH_SIZE 1

class Layer
{
private:
	friend class Ann;
	Tensor input;
	Tensor weights;
	Tensor output;
	Tensor bias;
	bool useBias;
	int outputSize, inputSize;
	
	float curBias;
public:
	Layer(int inputSize, int outputSize, bool useBias = false) :inputSize(inputSize), outputSize(outputSize), useBias(useBias)
	{
		for (int i = 0; i < inputSize; i++)
		{
			this->input.values().push_back(std::vector<float>());
			this->input.values()[i].resize(BATCH_SIZE);
		}
		this->input = this->input.T();


		this->weights.values().resize(1);
		for (int j = 0; j < this->inputSize; j++)
		{
				this->weights.values()[0].push_back(rx::Utility::randFloat(-50, 50) / static_cast<float>(100.f));
				if (std::abs(this->weights.values()[0][j]) < 0.1)
					this->weights.values()[0][j] *= 10.f;
		}
		this->weights = this->weights.T();

		this->bias.values().resize(this->outputSize);
		for (auto& e : this->bias.values())
		{
			e.push_back(rx::Utility::randFloat(-10, 10) / static_cast<float>(100.f));
		}
		this->bias = this->bias.T();

		this->curBias = rx::Utility::randFloat(-10, 10) / static_cast<float>(100.f);
	}
	void passInput(std::vector<std::vector<float>>& input)
	{
		this->input = input;
	}
	void passInput(Tensor &t )
	{
		this->input = t;
	}
	void forward(std::string actFun = "ReLU")
	{
		try
		{
			if (actFun != "ReLU" && actFun != "Sigmoid")
				throw std::runtime_error("Activation function \"" + actFun + "\" is not defined.\n");

		}
		catch (const std::exception& e)
		{
			std::cout << e.what();
			exit(1);
		}

		this->output = this->input * this->weights;
		this->output = this->output + 0.f;
		for (auto& e : this->output.values())
		{
			for (auto& k : e)
			{
				if(actFun == "ReLU")
					k = rx::Utility::ReLU(k);
				else
					k = rx::Utility::Sigmoid(k);

			}
		}
		if (this->useBias)
		{	
			this->output = this->output + this->curBias;
		}
		for (int i = 0; i < this->output.values().size(); i++)
		{
			for (int j = 1; j < this->outputSize; j++)
				this->output.values()[i].push_back(this->output.values()[i][0]);
		}
	}

	static void describe(Layer* layer)
	{
		if (!layer)
		{
			layer;
		}
	}
	void info()
	{
		std::cout << "In_features = " << this->input.values()[0].size()<<"\n";
		std::cout << "Out_features = " << outputSize;

	}
	void describe()
	{
		std::cout <<"Input:\n" << this->input.values();
		std::cout << "Weights:\n" << this->weights.values();
		std::cout << "Output:\n" << this->output.values(); 
		std::cout << "Bias :\n" << "[" << this->curBias << "]\n";
	}
	Tensor getOutput()
	{
		return this->output;
	}
	std::pair<int, int> features()
	{
		return std::make_pair(this->inputSize, this->outputSize);
	}
};

class Ann
{
private:
	std::vector<Layer*>layers;
	float learning_rate;
	float (*lossFun)(float z);

	Tensor input, y;
	Tensor grad;

public:
	Ann(){}

	void addLayer(int input, int output, bool bias = false);
	void forward(std::string input_actFun = "ReLU", std::string output_actFun = "Sigmoid");

	void describe();
	void info();
	static Tensor gradient(Tensor& input, Tensor& Error);
	
	void backProp();
	Tensor output();
	Tensor getGrad();
	Layer& getLayer(int index);
	void passValues(Tensor input, Tensor output);
	void setWeights(Layer& layer, Tensor weights);
	void setBias(Layer& layer, Tensor bias);
	
	std::vector<Layer*>& getLayers();

	
};


