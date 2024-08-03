#ifndef BNET_H
#define BNET_H

#include "Resources/Net.h"

#define BATCH_SIZE 1

class BNet : public Net
{
private:
	float gradi(float y, float yHat);
	Tensor grad_err(float y, Tensor yHat, Tensor &input, float grad, int depth);
	virtual void forward(std::string input_actFun = "ReLU", std::string output_actFun = "Sigmoid") override;
public:
	BNet() {};
	void train(int epochs = 1, bool debug = False, bool showAcc = False) override;
	void setLR(const float _learning_rate_) override;
	void backProp() override;
};
#endif