#ifndef CNET_H
#define CNET_H

#include "Resources/Net.h"
#define BATCH_SIZE 1

class CNet : public Net
{
private:
	std::map<int, std::vector<float>> outputs;

public:
	CNet() {};
	// void train(int epochs = 1, bool debug = False, bool showAcc = False) override;
	void setLR(const float _learning_rate_) override;
	void backProp() override;
	void test() { std::cout << "Test from CNet.\n"; }
};
#endif