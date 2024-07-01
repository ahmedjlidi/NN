#include "stdafx.h"
#include "Ann.h"

enum BOOL : bool
{
	True,
	False
};

void passInput(std::vector<std::vector<float>>& x, std::vector<float>& y, Ann& Model)
{
	Tensor input = Tensor(x);
	Tensor output = Tensor(y);
	Model.passValues(x, output);
}

void buildModel(Ann& Model)
{
	Model.addLayer(2, 4, True);
}

int main()
{
	std::vector<std::vector<float>> x = {  {1, 0} };
	std::vector<float> y = { 1 };
	Ann Model = Ann();
	buildModel(Model);
	passInput(x, y, Model);

	Model.forward();
	Model.getLayer(0).forward("Sigmoid");
	//Model.backProp();
	//std::cout << Model.getGrad().values();
	Model.describe();

	return 0;

}