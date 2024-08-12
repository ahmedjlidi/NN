#include "../stdafx.h"
#include "CNet.h"

void CNet::setLR(const float _learning_rate_)
{
	this->learning_rate = _learning_rate_;
	this->actFun_hidden = rx::Utility::ReLU;
	this->actFun_output = nullptr;
	this->param.actFun_h = "ReLU";
	this->param.actFun_o = "Softmax";
	this->param.lossFun = "Cross-entropy-loss";
	this->param.lr = _learning_rate_;

	const int size = this->layers[this->layers.size() - 1]->outputSize;
	for (int i = 0; i < size; i++)
	{
		std::vector<float> temp;
		temp.resize(size);
		temp[i] = 1;
		this->outputs[i] = temp;
	}
}

void CNet::backProp()
{
	for (int i = this->layers.size() - 1; i >= 0; i--)
	{

		Layer &layer = *this->layers[i];
		Tensor cur_error;
		// Backrop the output layer
		if (i == this->layers.size() - 1)
		{
			Tensor classifed_value;
			classifed_value.values().resize(1);
			classifed_value.values()[0] = this->outputs[this->y.values()[0][this->count]];
			cur_error = layer.output - classifed_value;
			layer.error.values().clear();
			layer.error = cur_error;

			Tensor T_input = layer.input.T();
			Tensor gradient = T_input * cur_error;
			gradient = gradient.T();

			if (i - 1 >= 0)
				this->layers[i - 1]->prev_weights = layer.weights;
			this->debug_parameters.weight_grad[i] = gradient;

			if (!this->avg_gradient[i].empty())
				this->avg_gradient[i] = this->avg_gradient[i] + gradient;
			else
			{
				this->avg_gradient[i] = gradient;
			}
			this->losses.push_back(rx::Utility::CrossEntropy(classifed_value.values(),
															 layer.getOutput().values()));
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
			Tensor T_error;
			if (i + 1 == this->layers.size() - 1)
				T_error = layer.error.T();
			else
				T_error = layer.error;
			Tensor T_weights = layer.prev_weights.T();
			cur_error = T_weights * T_error * dv_actFun_values;

			layer.error = cur_error;
			Tensor gradient = cur_error * layer.input;
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

				Tensor gradient = cur_error;
				this->debug_parameters.bias_grad[i] = gradient;

				if (!this->avg_bias[i].empty())
					this->avg_bias[i] = this->avg_bias[i] + gradient;
				else
					this->avg_bias[i] = gradient;
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