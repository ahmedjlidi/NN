#include "stdafx.h"
#include "Ann.h"


void passInput(std::vector<std::vector<float>>& x, std::vector<float>& y, Ann& Model)
{
	Tensor input = Tensor(x);
	Tensor output = Tensor(y);
	Model.passValues(x, output);
}

void buildModel(Ann& Model)
{
	Model.addLayer(2, 16 , true);
	Model.addLayer(16, 1, true);
}

void setWeights(Ann& Model)
{
	std::vector<float> t = { 0.1, 0.2};
	Tensor w = t;
	w = w.T();
	Model.setWeights(0, w);
}



int main()
{

	//And gate Dataset
	std::vector<std::vector<float>> x = {{1, 0}, { 0, 1 } , { 1, 1 }, {0,0} };
	std::vector<float> y = {1, 0, 0, 0};



	Ann Model = Ann();
	buildModel(Model);
	passInput(x, y, Model);
	Model.setParameters(0.1, "ReLU", "ReLU");
	Model.train(1000, True);


	//Model.describe();
	std::cout << Model.predict(x).values();
	std::cout << Ann::round(Model.predict(x), 0.5).values();

	return 0;

}