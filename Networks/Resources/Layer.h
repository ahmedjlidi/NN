#ifndef LAYER_H
#define LAYER_H

#include "../../Utility.h"
#include "../../Tensor.h"

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
	friend class Net;
	friend class BNet;
	friend class MAnn;

	Tensor input;
	Tensor weights;
	Tensor output;
	Tensor bias;
	Tensor weight_sum;
	bool useBias;
	int outputSize, inputSize;
	Tensor error;
	Tensor prev_weights;

	int getParamNum();

public:
	Layer(int inputSize, int outputSize, bool useBias = True);
	void passInput(std::vector<std::vector<float>> &input);
	void passInput(Tensor &t);
	void forward(std::string actFun = "ReLU");
	const bool usBias() const;

	void info();
	void describe();
	Tensor &getOutput();
	std::pair<int, int> features();
	void reset();
	Tensor &getWeights();

	void saveToFile();
	// void loadFromFile()
	// {
	// 	std::ifstream file("temp.txt", static_cast<std::ios_base::openmode>(std::ios::beg));
	// 	std::string str;
	// 	int count = 0;
	// 	while(std::getline(file, str))
	// 	{
	// 		if(str == "\n")
	// 		{
	// 			count++;
	// 		}
	// 		if(count == )
	// 	}
	// 	file << "\n";
	// 	file << "bias\n";
	// 	for(const auto& e: this->bias.values()) {
	// 		for(const auto& k: e) {
	// 			file << k<<" ";
	// 		}
	// 	}
	// 	file << "\n";
	// 	file.close();
	// }
};

#endif
