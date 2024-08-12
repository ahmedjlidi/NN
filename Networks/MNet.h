#ifndef MNET_H
#define MNET_H

#include "Resources/Net.h"

class MNet : public Net
{

public:
	MNet() {};
	void train(int epochs = 1, bool debug = False, bool showAcc = False) override;
	void setLR(const float _learning_rate_) override;
	void backProp() override;
};

#endif