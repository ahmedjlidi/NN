#include "../stdafx.h"
#include "MNet.h"

void MNet::setLR(const float _learning_rate_)
{
	this->learning_rate = _learning_rate_;
	this->actFun_hidden = rx::Utility::ReLU;
	this->actFun_output = nullptr;
	this->param.actFun_h = "ReLU";
	this->param.actFun_o = "None";
	this->param.lr = _learning_rate_;
	this->param.lossFun = "Mean-loss";
}

void MNet::backProp()
{
	for (int i = this->layers.size() - 1; i >= 0; i--)
	{

		Layer &layer = *this->layers[i];
		Tensor cur_error;
		// Backrop the output layer
		if (i == this->layers.size() - 1)
		{
			float error = rx::Utility::mse_dv(this->y.values()[0][this->count], layer.getOutput().values()[0][0]);
			layer.error.values().clear();
			layer.error = error;

			Tensor gradient = layer.input * error;

			if (i - 1 >= 0)
				this->layers[i - 1]->prev_weights = layer.weights;
			this->debug_parameters.weight_grad[i] = gradient;

			if (!this->avg_gradient[i].empty())
				this->avg_gradient[i] = this->avg_gradient[i] + gradient;
			else
			{
				this->avg_gradient[i] = gradient;
			}

			this->losses.push_back(
				rx::Utility::Mse(this->y.values()[0][this->count],
								 layer.getOutput().values()[0][0]));
		}
		// Backprop hidden layer
		else
		{
			Tensor dv_actFun_values;
			if (this->actFun_hidden == rx::Utility::ReLU)
			{
				dv_actFun_values.values() = rx::Utility::relu_dv(layer.weight_sum.values());
			}

			if (i + 1 >= this->layers.size() - 1)
			{
				layer.error = this->layers[this->layers.size() - 1]->error;
			}
			else
			{
				layer.error = this->layers[i + 1]->error;
			}

			dv_actFun_values = dv_actFun_values.T();
			Tensor T_weights = layer.prev_weights.T();
			cur_error = T_weights * layer.error * dv_actFun_values;

			layer.error = cur_error;
			Tensor T_input = layer.input;
			Tensor gradient = cur_error * T_input;
			this->debug_parameters.weight_grad[i] = gradient;
			if (i != 0)
				this->layers[i - 1]->prev_weights = layer.weights;

			if (!this->avg_gradient[i].empty())
				this->avg_gradient[i] = this->avg_gradient[i] + gradient;
			else
				this->avg_gradient[i] = gradient;
		}

		// //Backprop Bias
		if (layer.usBias())
		{
			// Bias for output layer
			if (i == this->layers.size() - 1)
			{

				float gradient = layer.error.values()[0][0];

				Tensor temp;
				temp.values().resize(1);
				temp.values()[0].push_back(gradient);
				this->debug_parameters.bias_grad[i] = temp;

				if (!this->avg_bias[i].empty())
					this->avg_bias[i] = this->avg_bias[i] + temp;
				else
					this->avg_bias[i] = temp;
			}
			else
			{
				Tensor gradient = cur_error;
				gradient = gradient.T();
				this->debug_parameters.bias_grad[i] = gradient;
				if (!this->avg_bias[i].empty())
					this->avg_bias[i] = this->avg_bias[i] + gradient;
				else
					this->avg_bias[i] = gradient;
			}
		}
	}
	this->currLoss = rx::Utility::mean(this->losses);
	this->losses.clear();
}

void MNet::train(int epochs, bool debug, bool showAcc)
{
	// Function to zero the gradients to prevent accumulation
	auto zero_grad = [this](std::map<int, Tensor> &grad_weight, std::map<int, Tensor> &grad_bias)
	{
		grad_weight.clear();
		grad_bias.clear();
	};

	// Check if epochs are less or equal to 0
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

	// Start training, each epoch finishes when done processing all samples in the dataset (Fpropagation, Backpropagation)
	const int maxE = epochs;
	while (epochs--)
	{
		// Use timer to measure epoch perfomance
		this->timer.start();
		float acc = 0.f;
		if (showAcc)
			acc = rx::Utility::accuracy(this->y.values(), Net::round(this->predict(this->input), 0.5).values());
		for (int i = 0; i < this->input.values().size(); i++)
		{
			// Forward propagation
			this->forward();

			// Zero the grads
			zero_grad(this->avg_gradient, this->avg_bias);

			// Calculate the loss and backpropagate
			this->backProp();

			// Average the gradients (batch gradient descent)
			auto weight_it = this->avg_gradient.begin();
			auto bias_it = this->avg_bias.begin();

			const float scaler = (1 / static_cast<float>(this->input.values().size()));
			while (weight_it != avg_gradient.end() && bias_it != avg_bias.end())
			{

				weight_it->second = weight_it->second * scaler;
				bias_it->second = bias_it->second * scaler;
				++weight_it;
				++bias_it;
			}

			// Update the weights (IF CODE WORKS DON'T TOUCH IT!!!!!)
			for (int i = 0; i < this->layers.size(); i++)
			{
				if (!this->avg_gradient[i].empty())
					updateWeights(this->layers[i]->weights, this->avg_gradient[i]);
				if (this->layers[i]->usBias() && !this->avg_bias[i].empty())
				{
					updateBias(this->layers[i]->bias, this->avg_bias[i]);
				}
			}

			// count is used to keep track of the current sample processed. Once reached the end reset it
			if (static_cast<unsigned long long>(this->count) + 1 >= this->input.values().size())
				this->count = 0;
			else
				this->count++;
		}

		// Print debug messages
		if (debug)
			printf("%d/%d epochs:------> Loss: %.3f  Time: %.3lf", maxE - epochs, maxE,
				   this->currLoss, this->timer.elapsed_time());
		if (showAcc)
			printf("Accuracy: %.2f", acc);
		printf("\n");
		this->timer.reset();
	}
}
