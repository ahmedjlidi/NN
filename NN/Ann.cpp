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

	Tensor temp = Tensor(this->input.values()[this->count]);
	this->layers[0]->passInput(temp);
	if(this->layers.size() == 1)
		this->layers[0]->forward(output_actFun);
	else
		this->layers[0]->forward(input_actFun);
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
	if (this->count + 1 >= this->input.values().size())
		this->count = 0;
	else
		this->count++;
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
		y = { y[this->count] };

		std::vector<float> err = rx::Utility::computeError(y, yHat);

		this->loss_grad = Tensor(err);

		Tensor in = Tensor(layer.input.values());
		Tensor out = Tensor(err);

		Tensor grad = Ann::gradient(out, in);
		
		this->grad = grad;
		grad = grad.T() * 0.01;

		//
		// cout << layer.weights.values() << grad.values();
		layer.weights = layer.weights - grad;
		layer.curBias = layer.curBias -  err[0] * 0.01;
		
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

void Ann::setBias(int index, float bias)
{
	this->layers[index]->curBias = bias;
}

void Ann::setWeights(int index, Tensor weights)
{
	this->layers[index]->weights = weights;

}

Tensor Ann::getGrad()
{
	return this->grad;
}

Tensor Ann::getLoss_grad()
{
	return this->loss_grad;
}

std::vector<Layer*>& Ann::getLayers()
{
	return this->layers;
}

Tensor Ann::predict(Tensor input, std::string input_actFun, std::string output_actFun)
{
	int c = 0;
	Tensor y;
	while (c < input.values().size())
	{
		Tensor temp = input.values()[c];
		this->layers[0]->passInput(temp);
		if (this->layers.size() == 1)
			this->layers[0]->forward(output_actFun);
		else
			this->layers[0]->forward(input_actFun);
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
		y.values().push_back(this->layers[this->layers.size() - 1]->output.values()[0]);
		c++;
	}


	return y;
	
}
