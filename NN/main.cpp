#include "stdafx.h"
#include "Ann.h"

enum BOOL : bool
{
	False,
	True
};

void passInput(std::vector<std::vector<float>>& x, std::vector<float>& y, Ann& Model)
{
	Tensor input = Tensor(x);
	Tensor output = Tensor(y);
	Model.passValues(x, output);
}

void buildModel(Ann& Model)
{
	Model.addLayer(2, 1 , True);
}

void setWeights(Ann& Model)
{
	std::vector<float> t = { 0.1, 0.2};
	Tensor w = t;
	w = w.T();
	Model.setWeights(0, w);
}

float gradi(float y, float yHat)
{
	yHat += 0.000001f;
	float v1 = (y * -1.f) / static_cast<float>(yHat);
	float v2 = (1 - y) / static_cast<float>(1 - yHat);
	return v1 + v2;
};
	
Tensor grad_err(float y, float yHat, Tensor& input, float grad)
{
	Tensor temp;
	temp.values().resize(1);
	for (int j = 0; j < input.values()[0].size(); j++)
	{
		temp.values()[0].push_back(yHat * (1 - yHat) * input.values()[0][j]);
		temp.values()[0][j] = temp.values()[0][j] * grad;
	}
	return temp;
}


void updateWeights(Tensor& w, Tensor n_w)
{
	n_w = n_w * 0.1;
	w = w - n_w;
}

std::vector<std::vector<float>> i = { {1, 0} };
std::vector<float> we = { 0.1, 0.2 };
std::vector<float> o = { 1 };
std::vector<float> oHat = { 0.5 };



int main()
{
	std::vector<std::vector<float>> x = { {1, 1} };
	std::vector<float> y = {0};
	Ann Model = Ann();
	buildModel(Model);
	setWeights(Model);
	passInput(x, y, Model);
	
	for (int i = 0; i < 100; i++)
	{
		Model.forward("ReLU", "Sigmoid");
		Model.backProp();
	}
	Model.describe();
	std::cout << Model.predict(x).values();




	return 0;

}

//Tensor x(i);
//Tensor y(o);
//Tensor yHat(oHat);
//Tensor weights(we);
//
//Tensor err = y - yHat;
//float loss = rx::Utility::loss(y.values()[0][0], yHat.values()[0][0]);
//float grad = gradi(y.values()[0][0], yHat.values()[0][0]);
//
////std::cout << loss << "\n" << grad << "\n";
////std::cout << grad_err(y.values()[0][0], yHat.values()[0][0], x, grad).values();
//updateWeights(weights, grad_err(y.values()[0][0], yHat.values()[0][0], x, grad));
//std::cout << weights.values() << "\n";