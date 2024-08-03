#include "../../stdafx.h"
#include "Layer.h"

int Layer::getParamNum()
{
	int total = 0;
	total += this->weights.getShape().first * this->weights.getShape().second;
	total += this->bias.getShape().first * this->bias.getShape().second;
	return total;
}
Layer::Layer(int inputSize, int outputSize, bool useBias) : inputSize(inputSize), outputSize(outputSize), useBias(useBias)
{
	this->input.values().resize(BATCH_SIZE);
	this->input.values()[0].resize(this->inputSize);

	float range = std::sqrt(static_cast<float>(1.f) / static_cast<float>(this->inputSize));
	for (int i = 0; i < this->outputSize; i++)
	{
		this->weights.values().push_back(std::vector<float>());
		for (int j = 0; j < this->inputSize; j++)
		{
			this->weights.values()[i].push_back(rx::Utility::kaiming_init(this->inputSize));
		}
	}
	this->bias.values().resize(1);
	this->bias.values()[0].resize(this->outputSize);
}

void Layer::passInput(std::vector<std::vector<float>> &input)
{
	this->input = input;
}

void Layer::passInput(Tensor &t)
{
	this->input = t;
}
void Layer::forward(std::string actFun)
{
	try
	{

		if (actFun != "ReLU" && actFun != "Sigmoid" && actFun != "None")
			throw std::runtime_error("Activation function \"" + actFun + "\" is defined.\n");
	}
	catch (const std::exception &e)
	{
		std::cout << e.what();
		exit(1);
	}
	Tensor tempweight = this->weights.T();
	this->output = this->input * tempweight;
	this->weight_sum = this->output;
	if (this->useBias)
	{
		this->output = this->output + this->bias;
	}
	if (actFun == "None")
	{
		return;
	}
	for (auto &e : this->output.values())
	{
		for (auto &k : e)
		{
			if (actFun == "ReLU")
				k = rx::Utility::ReLU(k);
			else
			{
				k = rx::Utility::Sigmoid(k);
			}
		}
	}
}
const bool Layer::usBias() const
{
	return this->useBias;
}

void Layer::info()
{
	std::cout << "In_features = " << this->input.values()[0].size() << "\n";
	std::cout << "Out_features = " << outputSize;
}
void Layer::describe()
{
	std::cout << "Input:\n"
			  << this->input.values();
	std::cout << "Weights :\n"
			  << this->weights.values();
	std::cout << "Output:\n"
			  << this->output.values();
	std::cout << "Bias :\n"
			  << this->bias.values();
}
Tensor &Layer::getOutput()
{
	return this->output;
}
std::pair<int, int> Layer::features()
{
	return std::make_pair(this->inputSize, this->outputSize);
}

void Layer::reset()
{
	this->input.values().resize(BATCH_SIZE);
	this->input.values()[0].resize(this->inputSize);

	float range = std::sqrt(static_cast<float>(1.f) / static_cast<float>(this->inputSize));
	for (int i = 0; i < this->outputSize; i++)
	{
		this->weights.values().push_back(std::vector<float>());
		for (int j = 0; j < this->inputSize; j++)
		{
			this->weights.values()[i].push_back(rx::Utility::randFloat(-10, 10) / static_cast<float>(100.f));
			if (std::abs(this->weights.values()[i][j]) < 0.1)
				this->weights.values()[i][j] *= 10.f;
		}
	}
	this->weights = this->weights.T();

	this->bias.values().resize(1);
	this->bias.values()[0].resize(this->outputSize);
}
Tensor &Layer::getWeights()
{
	return this->weights;
}

void Layer::saveToFile()
{
	std::ofstream file("temp.txt");
	for (const auto &e : this->weights.values())
	{
		for (const auto &k : e)
		{
			file << k << " ";
		}
	}
	file << "\n";
	for (const auto &e : this->bias.values())
	{
		for (const auto &k : e)
		{
			file << k << " ";
		}
	}
	file << "\n";
	file << "\n";
	file.close();
}