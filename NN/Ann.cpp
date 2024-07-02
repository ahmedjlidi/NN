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
	this->layers[this->layers.size() - 1]->curBias = 0.f;
}

void Ann::forward(std::string input_actFun, std::string output_actFun)
{
	if (this->param.actFun_h.empty() || this->param.actFun_o.empty())
	{
		std::cerr << "No hyper parameters were found.\n";
		exit(1);
	}
	Tensor temp = Tensor(this->input.values()[this->count]);
	this->layers[0]->passInput(temp);
	if(this->layers.size() == 1)
		this->layers[0]->forward(this->param.actFun_o);
	else
		this->layers[0]->forward(this->param.actFun_h);
	for (int i = 1; i < this->layers.size(); i++)
	{
		Tensor output = this->layers[i - 1]->getOutput();
		this->layers[i]->passInput(output);
		if (i + 1 >= this->layers.size())
			this->layers[i]->forward(this->param.actFun_o);
		else
		{
			this->layers[i]->forward(this->param.actFun_h);
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
	auto gradient = [](float y, float yHat)
		{
			yHat += 0.00001f;
			float v1 = (y * -1.f) / static_cast<float>(yHat);
			float v2 = (1 - y) / static_cast<float>(1 - yHat);

			return v1 + v2;
		};
	
	for (int i = this->layers.size() - 1; i >= 0; i--)
	{

		Layer& layer = *this->layers[i];

		
		float loss = rx::Utility::loss(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);
		if(i == this->layers.size() - 1)
			this->currLoss = loss;
		float gradient = gradi(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);
		updateWeights(layer.weights , grad_err(this->y.values()[0][this->count], layer.getOutput().values()[0][0],layer.input, gradient));
		
		auto bi_grad = [](float y, float yHat)
			{
				return  (y - yHat);
			};

		
		
		layer.curBias = layer.curBias +  this->learning_rate * bi_grad(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);
		
		this->grad = Tensor(grad_err(this->y.values()[0][this->count],
			layer.getOutput().values()[0][0], layer.input, gradient));

		std::vector<float> temp = { bi_grad(this->y.values()[0][this->count], layer.getOutput().values()[0][0]) };
		this->bi_grad = Tensor(temp);
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

Tensor Ann::getBI_grad()
{
	return this->bi_grad;
}

Tensor Ann::getLoss_grad()
{
	return this->loss_grad;
}

std::vector<Layer*>& Ann::getLayers()
{
	return this->layers;
}

Tensor Ann::round(Tensor t, float threshold)
{
	for(auto&e : t.values())
		for (auto& k : e)
		{
			if (k >= threshold)
			{
				k = std::ceil(k);
			}
			else
				k = std::floor(k);
		}

	return t;
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

void Ann::train(int epochs, bool debug)
{
	try 
	{
		if (epochs <= 0)
			throw std::runtime_error("Epochs should be >= 0.\n");
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
	int maxE = epochs;
	while (epochs--)
	{
		for (int i = 0; i < this->input.values().size(); i++)
		{
			this->forward();
			this->backProp();

			if (static_cast<unsigned long long>(this->count) + 1 >= this->input.values().size())
				this->count = 0;
			else
				this->count++;


			if (std::isnan(this->currLoss))
				this->currLoss = 0.f;
			printf("%d/%d epochs:______ Loss: %f\n", maxE - epochs, maxE, this->currLoss);
		}
	}
}

void Ann::setParameters(float lr, std::string actFun_hidden, std::string actFun_output)
{
	this->learning_rate = lr;
	if (actFun_hidden == "ReLU")
	{
		this->actFun_hidden = rx::Utility::ReLU;
	}
	else if (actFun_hidden == "Sigmoid")
	{
		this->actFun_hidden = rx::Utility::Sigmoid;
	}
	else
	{
		printf("%s is not defined activation function.\n", actFun_hidden.c_str());
		exit(1);
	}
	if (actFun_output == "ReLU")
	{
		this->actFun_output = rx::Utility::ReLU;
	}
	else if (actFun_output == "Sigmoid")
	{
		this->actFun_output = rx::Utility::Sigmoid;
	}
	else
	{
		printf("%s is not defined activation function.\n", actFun_output.c_str());
		exit(1);
	}
	this->param.actFun_h = actFun_hidden;
	this->param.actFun_o = actFun_output;
	this->param.lr = this->learning_rate;




}
