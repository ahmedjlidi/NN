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
	//this->layers[this->layers.size() - 1]->curBias = 0.f;
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

void Ann::describe(Ann& Model)
{
	std::cout << "\n-------------------------\n";
	for (const auto& e : Model.layers)
	{
		e->describe();
	}
	std::cout << "\n-------------------------\n";
}

void Ann::info(Ann& Model)
{
	std::cout << "\n-------------------------\n";
	for (const auto& e : Model.layers)
	{
		std::cout << "( ";
		e->info();
		std::cout << ")\n";
	}

	std::cout << "\nActivation function for hidden layers: ";
	if (Model.layers.size() > 1)
	{
		std::cout << Model.param.actFun_h << "\n";
	}
	else
		std::cout << "None.\n";
	std::cout << "Activation function for Output layer: "<< Model.param.actFun_o<<"\n";
	std::cout << "Loss function: " << Model.param.lossFun << "\n";
	std::cout << "Learning rate: " << Model.learning_rate << "\n";
	std::cout << "\n-------------------------\n";

}

void Ann::summary(Ann& Model)
{
	float total = 0;
	for (const auto& e : Model.layers)
	{
		total += (e->weights.values().size() * e->weights.values()[0].size()) * 4;
		total += 4;
	}
	std::string size;
	std::cout << "\n-------------------------\n";
	std::cout << "Parameters: " << total << "\n";
	if (total / static_cast<float>(1000000.f) < 0.1)
	{
		if (total / static_cast<float>(1000.f) < 0.1)
		{
			size = " Byte";
		}
		else
		{
			total /= static_cast<float>(1000.f);
			size = " Kb";
		}
	}
	else
	{
		total /= static_cast<float>(1000000.f);
		size = " Mb";
	}
	std::cout << "Size: " << total << size << "\n";
	std::cout << "In_shape: " << Model.layers[0]->input.values()[0].size() << "\n";
	std::cout << "Output_shape: " << Model.layers[Model.layers.size() - 1]->outputSize;
	std::cout << "\n-------------------------\n";
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
		auto err = [](Tensor yHat, float y)->Tensor
			{
				return yHat - y;
			};
		static Tensor error;
		float gradient = gradi(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);
		//Backrop the output layer
		if (i == this->layers.size() - 1)
		{
			float loss = rx::Utility::loss(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);
			
			Tensor g = grad_err(this->y.values()[0][this->count], layer.getOutput(), layer.input, gradient, i);
			//print(g.values(), 1);
			error = err(layer.getOutput().values(), this->y.values()[0][this->count]);
			updateWeights(layer.weights, g);
		}
		//Backprop hidden layer
		else
		{
			float dv_actfun = layer.getOutput().values()[0][0] > 0 ? 1 : 0;

			Tensor t;
			int lo_count = 0;
			for (const auto& e : layer.getOutput().values())
			{
				t.values().push_back(std::vector<float>());
				for (const auto& k : e)
				{
					t.values()[lo_count].push_back(k > 0 ? 1 : 0);
				}
			}
			Tensor temp;
			for(int h = 0; h < layer.weights.values().size(); h++)
			{
				std::vector<float>rowI = layer.weights.values()[h];
				Tensor weight_row(rowI);
				Tensor gr = (error * weight_row) * dv_actfun;
				temp.values().push_back(gr.values()[0]);
			}
			error = err(layer.getOutput().values(), this->y.values()[0][this->count]);
			updateWeights(layer.weights, temp);
		}

		
		//Backprop Bias
		if (layer.usBias())
		{
			auto bi_grad = [](float y, Tensor yHat)->Tensor
				{
					yHat = yHat * -1.f;
					return yHat + y;
				};
			if (i == this->layers.size() - 1)
			{
				float dv_loss = gradient * this->layers[this->layers.size() -1 ]->getOutput().values()[0][0] 
					* (1 - this->layers[this->layers.size() - 1]->getOutput().values()[0][0]);
				dv_loss = roundTo(dv_loss, 2);


				layer.bias = layer.bias - (this->learning_rate * dv_loss);
				//print(layer.bias.values(), 1);
			}
			else
			{
				//Tensor t;
				//t.values().resize(layer.getOutput().values().size());
				//for (int b = 0; b < layer.getOutput().values()[0].size(); b++)
				//{
				//	t.values()[0].push_back(layer.getOutput().values()[0][b]> 0 ? 1 : 0);
				//}
				//print(error.values(), 1);
				//t = t * error;
				////nprint(t.values(), 1);
			}
		}
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
	//this->layers[index]->curBias = bias;
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
		}
		if(debug)
			printf("%d/%d epochs:______ Loss: %f\n", maxE - epochs, maxE, this->currLoss);
	}
}

void Ann::compile(float lr, std::string actFun_hidden, std::string actFun_output)
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

void Ann::passData(const Tensor& x, const Tensor& y, Ann& Model)
{
	Model.passValues(x, y);
}
