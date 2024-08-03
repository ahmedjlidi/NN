#include "../stdafx.h"
#include "BNet.h"

float BNet::gradi(float y, float yHat)
{
	float v1 = yHat - y;
	float v2 = yHat * (1 - yHat);
	float output = v1 / static_cast<float>(v2);
	return roundTo(output, 4);
};
Tensor BNet::grad_err(float y, Tensor yHat, Tensor &input, float grad, int depth)
{
	Tensor temp;
	if (this->actFun_output == rx::Utility::Sigmoid)
	{
		temp.values() = rx::Utility::sigmoid_dv(input.values(), yHat.values());
	}
	else if (this->actFun_output == rx::Utility::ReLU)
	{
		temp.values() = rx::Utility::relu_dv(this->layers[depth]->weight_sum.values());
	}
	temp = temp * grad;
	return temp;
}
void BNet::forward(std::string input_actFun, std::string output_actFun)
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
void BNet::train(int epochs, bool debug, bool showAcc)
{
	try
	{
		if (epochs <= 0)
			throw std::runtime_error("Epochs should be >= 0.\n");
	}
	catch (const std::exception &e)
	{
		std::cout << e.what();
		exit(1);
	}
	int maxE = epochs;
	while (epochs--)
	{
		float acc;
		if (showAcc)
			acc = rx::Utility::accuracy(this->y.values(), BNet::round(this->predict(this->input), 0.5).T().values());
		else
			acc = -1;
		for (int i = 0; i < this->input.values().size(); i++)
		{
			this->forward();

			this->avg_gradient.clear();
			this->avg_bias.clear();
			this->backProp();
			for (auto &e : this->avg_gradient)
			{
				float scaler = (1 / static_cast<float>(this->input.values().size()));
				e.second = e.second * scaler;
			}
			for (int i = 0; i < this->layers.size(); i++)
			{
				updateWeights(this->layers[i]->weights, this->avg_gradient[i]);
			}
			for (auto &e : this->avg_bias)
			{
				float scaler = (1 / static_cast<float>(this->input.values().size()));
				e.second = e.second * scaler;
			}
			for (int i = 0; i < this->layers.size(); i++)
			{
				if (this->layers[i]->usBias())
					updateBias(this->layers[i]->bias, this->avg_bias[i]);
			}

			if (static_cast<unsigned long long>(this->count) + 1 >= this->input.values().size())
				this->count = 0;
			else
				this->count++;
		}
		if (debug)
			printf("%d/%d epochs:------> Loss: %.3f   ", maxE - epochs, maxE, this->currLoss);
		if (acc != -1)
			printf("Accuracy: %.2f", acc);
		printf("\n");
	}
}
void BNet::setLR(const float _learning_rate_)
{
	this->param.lossFun = "Binary-cross-entropy";
	this->learning_rate = _learning_rate_;
	this->actFun_hidden = rx::Utility::ReLU;
	this->actFun_output = rx::Utility::Sigmoid;
	this->param.actFun_h = "ReLU";
	this->param.actFun_o = "Sigmoid";
	this->param.lr = _learning_rate_;
}
void BNet::backProp()
{
	Layer prev_layer = *this->layers[this->layers.size() - 1];
	for (int i = this->layers.size() - 1; i >= 0; i--)
	{

		Layer &layer = *this->layers[i];
		auto err = [](Tensor yHat, float y) -> Tensor
		{
			return yHat - y;
		};

		static Tensor error;
		Tensor prev_Weights = layer.getWeights();
		float gradient = gradi(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);

		// Backrop the output layer
		if (i == this->layers.size() - 1)
		{
			float loss = rx::Utility::Bce(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);

			Tensor g = grad_err(this->y.values()[0][this->count], layer.getOutput(), layer.input, gradient, i);
			error = err(layer.getOutput().values(), this->y.values()[0][this->count]);
			this->layers[i - 1]->prev_weights = layer.weights;
			// updateWeights(layer.weights, g);
			this->debug_parameters.weight_grad[i] = g;

			if (!this->avg_gradient[i].empty())
				this->avg_gradient[i] = this->avg_gradient[i] + g;
			else
				this->avg_gradient[i] = g;

			this->currLoss = rx::Utility::Bce(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);
		}
		// Backprop hidden layer
		else
		{
			float dv_actfun = layer.getOutput().values()[0][0] > 0 ? 1 : 0;
			Tensor dv_actFun_values;
			int lo_count = 0;
			for (const auto &e : layer.weight_sum.values())
			{
				dv_actFun_values.values().push_back(std::vector<float>());
				for (const auto &k : e)
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
			if (i != 0)
				this->layers[i - 1]->prev_weights = layer.weights;
			// updateWeights(layer.weights, gradient);

			if (!this->avg_gradient[i].empty())
				this->avg_gradient[i] = this->avg_gradient[i] + gradient;
			else
				this->avg_gradient[i] = gradient;
		}

		// Backprop Bias
		if (layer.usBias())
		{
			// Bias for output layer
			auto bi_grad = [](float y, Tensor yHat) -> Tensor
			{
				yHat = yHat * -1.f;
				return yHat + y;
			};
			if (i == this->layers.size() - 1)
			{

				float dv_loss = gradient * this->layers[this->layers.size() - 1]->getOutput().values()[0][0] * (1 - this->layers[this->layers.size() - 1]->getOutput().values()[0][0]);
				dv_loss = roundTo(dv_loss, 4);

				Tensor temp;
				temp.values().resize(1);
				temp.values()[0].push_back(dv_loss);
				this->debug_parameters.bias_grad[i] = temp;

				if (!this->avg_bias[i].empty())
					this->avg_bias[i] = this->avg_bias[i] + temp;
				else
					this->avg_bias[i] = temp;

				// layer.bias = layer.bias - (this->learning_rate * dv_loss);
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

				if (!this->avg_bias[i].empty())
					this->avg_bias[i] = this->avg_bias[i] + gradient;
				else
					this->avg_bias[i] = gradient;
			}
		}

		if (i != this->layers.size() - 1)
		{
			error = err(layer.getOutput().values(), this->y.values()[0][this->count]);
		}
	}
}
