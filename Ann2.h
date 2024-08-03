#pragma once
#include "Timer.h"
#include "Networks/Resources/Layer.h"

class MAnn
{
private:
	std::vector<Layer *> layers;
	float learning_rate;
	float (*lossFun)(float z);
	float (*actFun_hidden)(float z);
	float (*actFun_output)(float z);

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

		DebugParam() {}
	};

	Timer timer;
	// Back prop temp values/////

	float currLoss;
	int count = 0;

	Tensor input, y;
	std::map<int, Tensor> avg_gradient;
	std::map<int, Tensor> avg_bias;

public:
	MAnn() {}

	// Setters////////////////////////////////////////
	void setWeights(int index, Tensor weights);
	void setBias(int index, float bias);
	///////////////////////////////////////////////

	void train(int epochs = 1, bool debug = False, bool showAcc = False);
	void compile(float lr, std::string actFun_hidden, std::string actFun_output);
	void debug(short type = ALL);
	void backProp();
	Tensor predict(Tensor input);

	// Getters//////////////////////////////////
	std::vector<Layer *> &getLayers();
	DebugParam debugParam();
	Layer &getLayer(int index);
	const float getCurrLoss();
	/////////////////////////////////////

	// static function////////////////////////////////
	static void describe(MAnn &Model);
	static void info(MAnn &Model);
	static void summary(MAnn &Model);
	static Tensor gradient(Tensor &input, Tensor &Error);
	static Tensor round(Tensor t, float threshold);
	static void passData(const Tensor &x, const Tensor &y, MAnn &Model);
	static void saveModel(MAnn &Model);
	// static void summary(Ann& Model);
	////////////////////////////////////////////////////

	void addLayer(int input, int output, bool bias = false);
	void forward(std::string input_actFun = "ReLU", std::string output_actFun = "Sigmoid");
	void setWeights(float _const_);

private:
	void updateWeights(Tensor &w, Tensor &n_w)
	{
		n_w = n_w * this->learning_rate;
		w = w - n_w;
	}
	void updateBias(Tensor &b, Tensor &n_b)
	{
		n_b = n_b * this->learning_rate;
		b = b - n_b;
	}

	void passValues(Tensor input, Tensor output);

	DebugParam debug_parameters;
	parameters param;

	// Functions which returns temp variables (for debug reasons)//
};