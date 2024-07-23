#pragma once
#include "Tensor.h"
#include "Utility.h"

#define BATCH_SIZE 1

enum BOOL : bool
{
	False,
	True
};


enum PARAM
{
	GRAD_WEIGHT,
	GRAD_BIAS,
	WEIGHT_SUM,
	OUTPUT,
	ALL
};

class Layer
{
private:
	friend class Ann;
	Tensor input;
	Tensor weights;
	Tensor output;
	Tensor bias;
	Tensor weight_sum;
	bool useBias;
	int outputSize, inputSize;
	Tensor error;
	Tensor prev_weights;
	

	int getParamNum()
	{
		int total = 0;
		total += this->weights.getShape().first * this->weights.getShape().second;
		total += this->bias.getShape().first * this->bias.getShape().second;
		return total;
	}
public:
	Layer(int inputSize, int outputSize, bool useBias = false) :inputSize(inputSize), outputSize(outputSize), useBias(useBias)
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
	void passInput(std::vector<std::vector<float>>& input)
	{
		this->input = input;
	}
	void passInput(Tensor &t )
	{
		this->input = t;
	}
	void forward(std::string actFun = "ReLU")
	{
		try
		{
			if (actFun != "ReLU" && actFun != "Sigmoid")
				throw std::runtime_error("Activation function \"" + actFun + "\" is not defined.\n");

		}
		catch (const std::exception& e)
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
		for (auto& e : this->output.values())
		{
			for (auto& k : e)
			{
				if(actFun == "ReLU")
					k = rx::Utility::ReLU(k);
				else
				{
					k = rx::Utility::Sigmoid(k);
				}
			}
		}

		
		
	}
	const bool usBias()
	{
		return this->useBias;
	}

	static void describe(Layer* layer)
	{
		if (!layer)
		{
			layer;
		}
	}
	void info()
	{
		std::cout << "In_features = " << this->input.values()[0].size()<<"\n";
		std::cout << "Out_features = " << outputSize;
	}
	void describe()
	{
		std::cout <<"Input:\n" << this->input.values();
		std::cout << "Weights :\n" << this->weights.values();
		std::cout << "Output:\n" << this->output.values(); 
		std::cout << "Bias :\n" << this->bias.values();
	}
	Tensor getOutput()
	{
		return this->output;
	}
	std::pair<int, int> features()
	{
		return std::make_pair(this->inputSize, this->outputSize);
	}


	void reset()
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
	Tensor& getWeights()
	{
		return this->weights;
	}


};



class Ann
{
private:
	std::vector<Layer*>layers;
	float learning_rate;
	float (*lossFun)(float z);
	float (*actFun_hidden)(float z);
	float (*actFun_output)(float z);

	Tensor input, y;
	std::map<int, Tensor> avg_gradient;
	std::map<int, Tensor> avg_bias;
	

	//Back prop temp values/////
	
	float currLoss;
	int count = 0;
	/////////////////////////////

	float currGrad;

	float gradi(float y, float yHat)
	{
		float v1 = yHat - y;
		float v2 = yHat * (1 - yHat);
		float output = v1 / static_cast<float>(v2);	
		return roundTo(output, 4);
	};
	Tensor grad_err(float y, Tensor yHat, Tensor& input, float grad, int depth)
	{
		Tensor temp;
		static float curr_loss = 0.f;
		for (int i = 0; i < yHat.values()[0].size(); i++)
		{
			temp.values().push_back(std::vector<float>());
			for (int j = 0; j < input.values()[0].size(); j++)
			{
				if (this->param.actFun_h == "ReLU" && depth < this->layers.size() - 1)
				{
					float val = yHat.values()[0][i] > 0 ? 1 : 0;
					temp.values()[i].push_back(val * input.values()[0][j]);
				}			
				else
				{
				

					temp.values()[i].push_back(yHat.values()[0][i] * (1 - yHat.values()[0][i]) * input.values()[0][j]);
				}

				temp.values()[i][j] = temp.values()[i][j] * grad;
			}
		}
		return temp;
	}
	void updateWeights(Tensor& w, Tensor n_w)
	{
		n_w = n_w * this->learning_rate;
		w = w - n_w;
	}
	void updateBias(Tensor& b, Tensor n_b)
	{
		n_b = n_b * this->learning_rate;
		b = b - n_b;
	}


	struct parameters
	{
		std::string actFun_h;
		std::string actFun_o;
		std::string lossFun = "Binary-cross entropy";
		float lr;
	};

	struct DebugParam
	{
		std::map<int, Tensor> weight_grad;
		std::map<int, Tensor> bias_grad;
		std::map<int, Tensor> weight_sum;
		std::map<int, Tensor> a_hidden;

		DebugParam(){}
	};

	DebugParam debug_parameters;
	parameters param;
	void passValues(Tensor input, Tensor output);
public:
	Ann(){}

	void addLayer(int input, int output, bool bias = false);
	void forward(std::string input_actFun = "ReLU", std::string output_actFun = "Sigmoid");
	void setWeights(float _const_);
	
	

	//Functions which returns temp variables (for debug reasons)//

	
	


	
	
	
	//Setters////////////////////////////////////////
	void setWeights(int index, Tensor weights);
	void setBias(int index, float bias);
	///////////////////////////////////////////////
	
	
	void train(int epochs = 1, bool debug = False, bool showAcc = False);
	void compile(float lr, std::string actFun_hidden, std::string actFun_output);
	void debug(short type = ALL);
	void backProp();
	Tensor predict(Tensor input, std::string input_actFun = "ReLU", std::string output_actFun = "Sigmoid");



	//Getters//////////////////////////////////
	std::vector<Layer*>& getLayers();
	DebugParam debugParam();
	Layer& getLayer(int index);
	const float getCurrLoss();
	/////////////////////////////////////
	
	//static function////////////////////////////////
	static void describe(Ann& Model);
	static void info(Ann& Model);
	static void summary(Ann& Model);
	static Tensor gradient(Tensor& input, Tensor& Error);
	static Tensor round(Tensor t, float threshold);
	static void passData(const Tensor& x, const Tensor& y, Ann& Model);
	static void saveModel(Ann& Model);
	//static void summary(Ann& Model);
	////////////////////////////////////////////////////

	

	
	
};




