#include "stdafx.h"
#include "Ann.h"

void setWeights(Ann& Model)
{
	std::vector<std::vector<float>> v1 = { {0.1, 0.1}, {0.1, 0.1} };
	std::vector<std::vector<float>> v2 = { {0.1, 0.1} };
	Model.setWeights(0, Tensor(v1));
	Model.setWeights(1, Tensor(v2));
}

int main()
{

	//And gate Dataset
	std::vector<std::vector<float>> x = { {1, 1}, {0, 0}, {1, 0}, {0, 1} };
	std::vector<float> y = { 0, 0, 1, 1};
	Tensor labels(y);


	Ann Model = Ann();
	Model.addLayer(2, 8, True);
	Model.addLayer(8, 1, True);
	//setWeights(Model);
	Ann::passData(x, y, Model);
	Model.compile(0.01, "ReLU", "Sigmoid");

	//Model.train(500, True);
	Model.train(500);

	
	//Ann::describe(Model);
	print(Model.predict(x).values());
	/*std::cout << Ann::round(Model.predict(x), 0.5).values();
	std::cout << "Accuracy: "<< 
		rx::Utility::accuracy(labels.values(), Ann::round(Model.predict(x), 0.5).T().values()) <<"%\n";*/

	return 0;


}

