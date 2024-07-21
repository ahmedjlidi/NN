#include "stdafx.h"
#include "Ann.h"

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
	if (this->param.actFun_h.empty() || this->param.actFun_o.empty())
	{
		std::cerr << "No hyper parameters were found.\n";
		exit(1);
	}
	Tensor temp = Tensor(this->input.values()[this->count]);
	this->layers[0]->passInput(temp);
	if (this->layers.size() == 1)
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
	std::cout << "Activation function for Output layer: " << Model.param.actFun_o << "\n";
	std::cout << "Loss function: " << Model.param.lossFun << "\n";
	std::cout << "Learning rate: " << Model.learning_rate << "\n";
	std::cout << "\n-------------------------\n";

}

void Ann::summary(Ann& Model)
{
	printf("\n--------------------------------------------------------\n"
		"	Layer (type)		Output Shape	Param #\n"
		"========================================================\n");
	for (int i = 0; i < Model.layers.size(); i++)
	{
		printf("	Layer %d:\t\t[%d, %d]\t\t%d\n", i, 1, Model.layers[i]->outputSize, Model.layers[i]->getParamNum());
	}
	printf("========================================================\n");


	float total = 0, train_total = 0.f, total_size = 0.f;

	for (const auto& e : Model.layers)
	{
		//Size in bytes
		total_size += (e->weights.values().size() * e->weights.values()[0].size()) * 4;
		total_size += (e->bias.getShape().first * e->bias.getShape().second) * 4;

		//trainable params
		train_total += (e->weights.values().size() * e->weights.values()[0].size());
		if (e->useBias)
		{
			train_total += (e->bias.getShape().first * e->bias.getShape().second);
		}

		total += (e->weights.values().size() * e->weights.values()[0].size());
		total += (e->bias.getShape().first * e->bias.getShape().second);
	}
	std::string size;
	std::cout << "\n-------------------------\n";
	std::cout << "Total Parameters: " << total << "\n";
	std::cout << "Trainable Parameters: " << train_total << "\n";
	std::cout << "Non Trainable Parameters: " << total - train_total << "\n";
	if (total_size / static_cast<float>(1000000.f) < 0.1)
	{
		if (total_size / static_cast<float>(1000.f) < 0.1)
		{
			size = " Byte";
		}
		else
		{
			total_size /= static_cast<float>(1000.f);
			size = " Kb";
		}
	}
	else
	{
		total_size /= static_cast<float>(1000000.f);
		size = " Mb";
	}
	std::cout << "Parameters Size: " << total_size << size << "\n";
	std::cout << "Input Shape: " << Model.layers[0]->input.values()[0].size() << "\n";
	std::cout << "Output shape: " << Model.layers[Model.layers.size() - 1]->outputSize;
	std::cout << "\n-------------------------\n";
}

Tensor Ann::gradient(Tensor& input, Tensor& Error)
{
	return input * Error;
}

void Ann::backProp()
{
	Layer prev_layer = *this->layers[this->layers.size() - 1];
	for (int i = this->layers.size() - 1; i >= 0; i--)
	{

		Layer& layer = *this->layers[i];
		auto err = [](Tensor yHat, float y)->Tensor
			{
				return yHat - y;
			};

		static Tensor error;
		Tensor prev_Weights = layer.getWeights();
		float gradient = gradi(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);

		//Backrop the output layer
		if (i == this->layers.size() - 1)
		{
			float loss = rx::Utility::loss(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);

			Tensor g = grad_err(this->y.values()[0][this->count], layer.getOutput(), layer.input, gradient, i);
			error = err(layer.getOutput().values(), this->y.values()[0][this->count]);
			this->layers[i - 1]->prev_weights = layer.weights;
			//updateWeights(layer.weights, g);
			this->debug_parameters.weight_grad[i] = g;


			if(!this->avg_gradient[i].empty())
				this->avg_gradient[i] = this->avg_gradient[i] + g;
			else
				this->avg_gradient[i] = g;

			this->currLoss = rx::Utility::loss(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);
		}
		//Backprop hidden layer
		else
		{
			float dv_actfun = layer.getOutput().values()[0][0] > 0 ? 1 : 0;
			Tensor dv_actFun_values;
			int lo_count = 0;
			for (const auto& e : layer.weight_sum.values())
			{
				dv_actFun_values.values().push_back(std::vector<float>());
				for (const auto& k : e)
				{
					dv_actFun_values.values()[lo_count].push_back(k > 0 ? 1 : 0);
				}
			}
			if (i + 1 >= this->layers.size() - 1)
			{
				layer.error = error;
			}
			else
			{
				layer.error = this->layers[i + 1]->error;
			}
		
			dv_actFun_values = dv_actFun_values.T();
			Tensor T_weights = layer.prev_weights.T();
			Tensor cur_error = T_weights * layer.error * dv_actFun_values;
			
			
			layer.error = cur_error;
			Tensor T_input = layer.input;
			Tensor gradient = cur_error * T_input;
			this->debug_parameters.weight_grad[i] = gradient;
			if(i != 0)
				this->layers[i - 1]->prev_weights = layer.weights;
			//updateWeights(layer.weights, gradient);

			if (!this->avg_gradient[i].empty())
				this->avg_gradient[i] = this->avg_gradient[i] + gradient;
			else
				this->avg_gradient[i] = gradient;
		}


		//Backprop Bias
		if (layer.usBias())
		{
			//Bias for output layer
			auto bi_grad = [](float y, Tensor yHat)->Tensor
				{
					yHat = yHat * -1.f;
					return yHat + y;
				};
			if (i == this->layers.size() - 1)
			{

				float dv_loss = gradient * this->layers[this->layers.size() - 1]->getOutput().values()[0][0]
					* (1 - this->layers[this->layers.size() - 1]->getOutput().values()[0][0]);
				dv_loss = roundTo(dv_loss, 4);
				
				Tensor temp; temp.values().resize(1);temp.values()[0].push_back(dv_loss);
				this->debug_parameters.bias_grad[i] = temp;


				if (!this->avg_gradient[i].empty())
					this->avg_gradient[i] = this->avg_gradient[i] + g;
				else
					this->avg_gradient[i] = g;


				layer.bias = layer.bias - (this->learning_rate * dv_loss);
				
			}
			else
			{
				Tensor dv_act_fun;
				dv_act_fun.values().resize(layer.weight_sum.values().size());
				for (int b = 0; b < layer.weight_sum.values()[0].size(); b++)
				{
					dv_act_fun.values()[0].push_back(layer.getOutput().values()[0][b] > 0 ? 1 : 0);
				}
				dv_act_fun = dv_act_fun.T();
				Tensor T_weight = layer.prev_weights.T();
				Tensor curr_error;
				if (i + 1 >= this->layers.size() - 1)
				{
					curr_error = error;
				}
				else
				{
					curr_error = this->layers[i + 1]->error;
				}
				Tensor gradient = T_weight * curr_error * dv_act_fun;
				gradient = gradient.T();
				this->debug_parameters.bias_grad[i] = gradient;
				gradient = gradient * this->learning_rate;

				layer.bias = layer.bias - gradient;
				prev_layer = *this->layers[i + 1];

				
			}
		}

		if (i != this->layers.size() - 1)
		{
			error = err(layer.getOutput().values(), this->y.values()[0][this->count]);
		}
	}
}

Tensor Ann::output()
{
	Tensor t = this->layers[this->layers.size() - 1]->output;
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
	//this->debug_parameters = Ann::DebugParam(this->input.values().size());

}

void Ann::setWeights(int index, Tensor weights)
{
	this->layers[index]->weights = weights;
}

std::vector<Layer*>& Ann::getLayers()
{
	return this->layers;
}

Tensor Ann::round(Tensor t, float threshold)
{
	for (auto& e : t.values())
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
		float acc = rx::Utility::accuracy(this->y.values(), Ann::round(this->predict(this->input), 0.5).T().values());
		for (int i = 0; i < this->input.values().size(); i++)
		{
			this->forward();

			this->avg_gradient.clear();
			this->backProp();
			for (auto& e : this->avg_gradient)
			{
				float scaler = (1 / static_cast<float>(this->input.values().size()));
				e.second= e.second * scaler;
			}
			for (int i = 0; i < this->layers.size(); i++)
			{
				updateWeights(this->layers[i]->weights, this->avg_gradient[i]);
			}
			

			if (static_cast<unsigned long long>(this->count) + 1 >= this->input.values().size())
				this->count = 0;
			else
				this->count++;

		}
		if (debug)
			printf("%d/%d epochs:------> Loss: %.3f   Accuracy: %f \n", maxE - epochs, maxE, this->currLoss, acc);
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

Ann::DebugParam Ann::debugParam()
{
	return this->debug_parameters;
}

void Ann::debug(short type)
{

	auto printWeight_grad = [this]()
		{
			for (const auto& e : this->debug_parameters.weight_grad)
			{
				Tensor temp = e.second;
				printf("Weight gradient for layer %d:\n", e.first);
				print(temp.values(), 1);
			}
		};
	auto printBias_grad = [this]() {


		for (const auto& e : this->debug_parameters.bias_grad)
		{
			Tensor temp = e.second;
			printf("Gradient of bias for layer %d:\n", e.first);
			print(temp.values(), 1);
		}
		};
	auto printWeight_sum = [this]() {
		for (const auto& e : this->debug_parameters.weight_sum)
		{
			Tensor temp = e.second;
			printf("Weight sum for layer %d:\n", e.first);
			print(temp.values(), 1);
		}
		};
	auto printActiv_value = [this]() {
		for (const auto& e : this->debug_parameters.a_hidden)
		{
			Tensor temp = e.second;
			printf("Activated values for layer %d:\n", e.first);
			print(temp.values(), 1);
		}
		};

	print("\n=============================\n");
	switch (type)
	{
	case ALL:
		printWeight_grad();
		print("********************\n");
		printBias_grad();
		print("********************\n");
		printWeight_sum();
		print("********************\n");
		printActiv_value();
		print("********************\n");
		break;
	case WEIGHT_SUM:
		printWeight_sum();
		break;
	case GRAD_BIAS:
		printBias_grad();
		break;
	case GRAD_WEIGHT:
		printWeight_grad();
		break;
	case OUTPUT:
		printActiv_value();
		break;
	default:
		printf("No such option as %d\n", type);
		return;
	}
	print("\n=============================\n");
}
