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
	Model.addLayer(2, 16, True);
	Model.addLayer(16, 1, True);
}

void setWeights(Ann& Model)
{
	std::vector<float> t = { -0.2735, -0.2303 };
	Tensor w = t;
	w = w.T();
	Model.setWeights(0, w);
	Model.setBias(0, -0.0611);
}
int main()
{
	std::vector<std::vector<float>> x = { {1, 0}, {1, 1}, {0, 1},  {0, 0} };
	std::vector<float> y = { 1 ,0, 1, 0};
	Ann Model = Ann();
	buildModel(Model);
	//setWeights(Model);
	passInput(x, y, Model);
	
	for (int i = 0; i < 1000 ; i++)
	{
		Model.forward();
		Model.backProp();
	}
	//Model.describe();
	std::cout << Model.output().values();
	std::cout << Model.predict(x).values();
	//Model.describe();


	/*std::cout << Model.getGrad().values()<<"\n";
	std::cout << Model.getLoss_grad().values();*/

	/*std::vector<float> err{ 0.5 };
	std::vector<float> input = { {1, 0} };
	std::vector<float> weights = { {0.1, 0.2} };
	
	Tensor in(input);
	Tensor out(err);
	Tensor weight(weights);

	Tensor grad = Ann::gradient(out, in);
	grad = grad * 0.01;
	weight = weight - grad;
	std::cout << grad.values();
	std::cout << weight.values() << "\n*/

	return 0;

}