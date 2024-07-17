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
	Layer next_layer = prev_layer;
	for (int i = this->layers.size() - 1; i >= 0; i--)
	{
		
		Layer& layer = *this->layers[i];
		this->debug_parameters.weight_sum[i] = layer.weight_sum;
		this->debug_parameters.a_hidden[i] = layer.getOutput();

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
			layer.error = error;
			updateWeights(layer.weights, g);
			debug_parameters.weight_grad[i] = g;
			this->currLoss = rx::Utility::loss(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);
		}
		//Backprop hidden layer
		else
		{
			if (i != 0)
				next_layer = *this->layers[i - 1];
			if (i + 1 != this->layers.size() - 1)
			{
				prev_layer = *this->layers[i + 1];
			}
			Tensor dv_actfun;
			dv_actfun.values().resize(layer.getOutput().values().size());
			for (int b = 0; b < layer.getOutput().values()[0].size(); b++)
			{
				dv_actfun.values()[0].push_back(layer.getOutput().values()[0][b] > 0 ? 1 : 0);
			}

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
			/*for(int h = 0; h < layer.weights.values().size(); h++)
			{
				std::vector<float>rowI = layer.weights.values()[h];
				Tensor weight_row(rowI);
				Tensor gr;
				if (error.getShape().first == weight_row.getShape().first
					&& error.getShape().second == weight_row.getShape().second)
				{
					for (int z = 0; z < error.getShape().first; z++)
					{
						gr.values().push_back(std::vector<float>());
						for (int y = 0; y < error.getShape().second; y++)
						{
							gr.values()[z].push_back(error.values()[z][y] * weight_row.values()[z][y]);
						}
					}
				}
				else
				{
					gr = error * weight_row;
				}
				gr = gr * dv_actfun;
				temp.values().push_back(gr.values()[0]);

			}*/

			Tensor hid_error = this->layers[this->layers.size() -1]->error;
			for (int l = 0; l < prev_layer.outputSize - 1; l++)
			{
				hid_error.values()[0].push_back(hid_error.values()[0][0]);
			}
			//print(hid_error.values());
			//print(hid_error.values());
			hid_error = hid_error.T();
			
			if (i != 0)
			{
				temp = hid_error * prev_layer.weights;
				temp = temp.T();
				temp = temp * next_layer.output;
				//print(temp.values());
				
				
			}
			else
			{
				Tensor temp_output;
				for (int h = 0; h < prev_layer.weights.values().size(); h++)
				{
					temp_output.values().push_back(next_layer.output.values()[0]);
				}

				temp = prev_layer.weights * hid_error.values()[0][0];
				temp = temp.T();
				temp = temp * temp_output;
					
			}

			updateWeights(layer.weights, temp);
			debug_parameters.weight_grad[i] = temp;
			
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
				
				float dv_loss = gradient * this->layers[this->layers.size() -1 ]->getOutput().values()[0][0] 
					* (1 - this->layers[this->layers.size() - 1]->getOutput().values()[0][0]);
				dv_loss = roundTo(dv_loss, 2);


				layer.bias = layer.bias - (this->learning_rate * dv_loss);
				debug_parameters.bias_grad[i] = dv_loss;
			}
			else
			{

				Tensor dv_act_fun;
				dv_act_fun.values().resize(layer.getOutput().values().size());
				for (int b = 0; b < layer.getOutput().values()[0].size(); b++)
				{
					dv_act_fun.values()[0].push_back(layer.getOutput().values()[0][b]> 0 ? 1 : 0);
				}
				
				Tensor delta_hidden;
				delta_hidden= error * prev_layer.weights;
				Tensor mult;
				if (dv_act_fun.getShape().first == delta_hidden.getShape().first
					&& dv_act_fun.getShape().second == delta_hidden.getShape().second)
				{
					for (int z = 0; z < delta_hidden.getShape().first; z++)
					{
						mult.values().push_back(std::vector<float>());
						for (int y = 0; y < delta_hidden.getShape().second; y++)
						{
							mult.values()[z].push_back(dv_act_fun.values()[z][y] * delta_hidden.values()[z][y]);
						}
					}
				}
				else
				{
					mult = delta_hidden * dv_act_fun;
				}

				
				delta_hidden = mult;
				/*print("-------\ngradient of bias : ");
				print(delta_hidden.values());*/
				delta_hidden = delta_hidden * this->learning_rate;
				layer.bias = layer.bias - delta_hidden;
				/*print("New bias : ");
				print(layer.bias.values());
				print("------\n");*/
				
				
				debug_parameters.bias_grad[i] = delta_hidden;
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
		float acc = rx::Utility::accuracy(this->y.values(), Ann::round(this->predict(this->input), 0.5).T().values());
		for (int i = 0; i < this->input.values().size(); i++)
		{
			this->forward();
			this->backProp();

			if (static_cast<unsigned long long>(this->count) + 1 >= this->input.values().size())
				this->count = 0;
			else
				this->count++;
		}
		if(debug)
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
