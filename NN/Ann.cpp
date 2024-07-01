#include "stdafx.h"
#include "Ann.h"

void Ann::addLayer(int input, int output, bool bias)
{
	try
	{
		if (input <= 0 || output <= 0)
			throw std::runtime_error("Input and output should be >= 0.\n");
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
	this->layers.push_back(new Layer(input, output, bias));
}

void Ann::forward(std::string input_actFun, std::string output_actFun)
{

	for (auto it = this->layers.begin(); it != this->layers.end(); ++it)
	{
		auto& layer = *it;
		if (it + 1 >= this->layers.end())
		{
			layer->forward(output_actFun);
		}
		else
		{
			layer->forward(input_actFun);
		}
	}

	this->layers[0]->passInput(this->input);
	this->layers[0]->forward();
	for (int i = 1; i < this->layers.size(); i++)
	{
		Tensor output = this->layers[i - 1]->getOutput();
		this->layers[i]->passInput(output);
		if (i + 1 >= this->layers.size())
			this->layers[i]->forward(output_actFun);
		else
		{
			this->layers[i]->forward(input_actFun);
		}
	}
}

void Ann::describe()
{
	for (const auto& e : this->layers)
	{
		e->describe();
	}
}

void Ann::info()
{
	for (const auto& e : this->layers)
	{
		e->info();
	}
}

Tensor Ann::gradient(Tensor& input, Tensor& Error)
{
	return input * Error;
}

void Ann::backProp()
{
	for (int i = this->layers.size() - 1; i >= 0; i--)
	{
		Layer& layer = *this->layers[i];

		std::vector<float> yHat = layer.getOutput().T().squeeze();
		std::vector<float> y = this->y.squeeze();

		std::vector<float> err = rx::Utility::computeError(y, yHat);

		Tensor in = Tensor(layer.input.values());
		Tensor out = Tensor(err);

		in = in.T();
		out = out.T();

		Tensor grad = Ann::gradient(in, out);
		this->grad = grad;
		grad = grad * 0.1;
		layer.weights = layer.weights - grad;

		

		/*Tensor bi = out * -0.1;
		bi = bi.T();
		std::vector<float> vals;
		for (int i = 0; i < bi.values().size(); i++)
		{
			vals.push_back(rx::Utility::sum(bi.values()[i]));
		}*/
		//std::cout << "---------\n" << vals << "\n\n" << layer.bias.values() << "---------\n";
		/*std::cout << "-----\n";
		std::cout << bi.values() << "\n\n" << b << "\n";
		std::cout << "-----\n";
		b.clear();*/
		
		//layer.bias = layer.bias.colAdd(vals);
		
	}
}

Tensor Ann::output()
{
	Tensor t  = this->layers[this->layers.size() - 1]->output;
	return t;
}

Layer& Ann::getLayer(int index)
{
	return *this->layers[index];
}

void Ann::passValues(Tensor input, Tensor output)
{
	this->input = Tensor(input.values());
	this->y = output;
}

void Ann::setBias(Layer& layer, Tensor bias)
{
	layer.bias = bias;
}

void Ann::setWeights(Layer& layer, Tensor weights)
{
	layer.input = weights;

}

Tensor Ann::getGrad()
{
	return this->grad;
}

std::vector<Layer*>& Ann::getLayers()
{
	return this->layers;
}


