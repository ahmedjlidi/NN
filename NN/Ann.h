#pragma once
#include "Tensor.h"
#include "Utility.h"

#define BATCH_SIZE 1

enum PARAM
{
	INPUT,
	WEIGHTS,
	BIAS,
	OUTPUT,
	ALL
};

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
	float kaimingInit(int fanIn) 
	{
		// Create a random device and Mersenne Twister generator
		std::random_device rd;
		std::mt19937 gen(rd());

		// Standard deviation for Kaiming initialization
		float stddev = std::sqrt(2.0f / fanIn);

		// Create a normal distribution with mean 0 and calculated stddev
		std::normal_distribution<float> d(0.0f, stddev);

		// Generate and return a random value
		return d(gen);
	}
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

		float range = static_cast<float>(1.f) / static_cast<float>(this->inputSize);
		for (int j = 0; j < this->inputSize; j++)
		{
				this->weights.values()[0].push_back(rx::Utility::randFloat(-100, 100) / static_cast<float>(100.f));
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

		this->curBias = 0.f;
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
		for (auto& e : this->output.values())
		{
			for (auto& k : e)
			{
				if(actFun == "ReLU")
					k = rx::Utility::ReLU(k);
				else
				{
					k = rx::Utility::Sigmoid(k);
				}
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
		std::cout << "Weights :\n" << this->weights.values();
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

	void clear()
	{
		//this->input.values().clear();
		//this->output.values().clear();
	}

	/*Tensor parameters(short type = ALL)
	{
		switch (type)
		{
		case ALL:
			
			
			
			this->w
		}
	}*/

	

	
};

class Ann
{
private:
	std::vector<Layer*>layers;
	float learning_rate;
	float (*lossFun)(float z);

	Tensor input, y;
	

	//Back prop temp values/////
	Tensor grad;
	Tensor loss_grad;
	int count = 0;
	std::vector<float> losses;
	/////////////////////////////



	float gradi(float y, float yHat)
	{
		yHat += 0.000005f;
		float v1 = (y * -1.f) / static_cast<float>(yHat);
		float v2 = (1 - y) / static_cast<float>(1 - yHat);
		return v1 + v2;
	};
	Tensor grad_err(float y, float yHat, Tensor& input, float grad)
	{
		Tensor temp;
		temp.values().resize(1);
		for (int j = 0; j < input.values()[0].size(); j++)
		{
			temp.values()[0].push_back(yHat * (1 - yHat) * input.values()[0][j]);
			temp.values()[0][j] = temp.values()[0][j] * grad;
		}
		return temp;
	}
	void updateWeights(Tensor& w, Tensor n_w)
	{
	
		n_w = n_w * 0.1;
		n_w = n_w.T();
		w = w - n_w;
	}

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
	Tensor getLoss_grad();
	Layer& getLayer(int index);
	void passValues(Tensor input, Tensor output);
	void setWeights(int index, Tensor weights);
	void setBias(int index, float bias);
	Tensor predict(Tensor input, std::string input_actFun = "ReLU", std::string output_actFun = "Sigmoid");
	std::vector<Layer*>& getLayers();

};


