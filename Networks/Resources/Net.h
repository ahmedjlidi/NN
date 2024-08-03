#ifndef NET_H
#define NET_H

#include "Layer.h"
#include "../../Timer.h"

class Net
{
	friend class BNet;
protected:
	std::vector<Layer *> layers;
	float learning_rate;
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

	void updateWeights(Tensor &w, Tensor &n_w);
	void updateBias(Tensor &b, Tensor &n_b);

	void passValues(Tensor input, Tensor output);

	DebugParam debug_parameters;
	parameters param;

public:
	Net() {}
	virtual ~Net();

	// Setters////////////////////////////////////////
	virtual void setWeights(int index, Tensor weights);
	virtual void setBias(const float _const_);
	virtual void setLR(const float _learning_rate_);
	virtual void setWeights(const float _const_);
	virtual void addLayer(int input, int output, bool bias = true);
	///////////////////////////////////////////////

	///////////////////////////////////////////////
	virtual void forward(std::string input_actFun = "ReLU", std::string output_actFun = "Sigmoid");
	virtual void backProp() = 0;
	virtual void train(int epochs = 1, bool debug = False, bool showAcc = False);
	//////////////////////////////////////////////////////////

	// Getters//////////////////////////////////
	virtual std::vector<Layer *> &getLayers();
	virtual DebugParam debugParam();
	virtual Layer &getLayer(int index);
	virtual const float getCurrLoss();
	/////////////////////////////////////

	// static function////////////////////////////////
	static void describe(Net &Model);
	static void info(Net &Model);
	static void summary(Net &Model);
	static Tensor round(Tensor t, float threshold);
	static void passData(const Tensor &x, const Tensor &y, Net &Model);
	static void saveModel(Net &Model);
	virtual void debug(short type = ALL);
	// static void summary(Ann& Model);
	////////////////////////////////////////////////////

	virtual Tensor predict(Tensor input);
};
#endif
