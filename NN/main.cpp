#include "stdafx.h"
#include "Ann.h"



int main()
{

	//And gate Dataset
	std::vector<std::vector<float>> x = {{1, 0}, { 0, 1 } , { 1, 1 }, {0,0} };
	std::vector<float> y = {1, 0, 0, 0};
	Tensor labels(y);


	Ann Model = Ann();
	Model.addLayer(2, 8, True);
	Model.addLayer(8, 1, True);
	Ann::passData(x, y, Model);
	Model.compile(0.1, "ReLU", "Sigmoid");

	
	Model.train(100, True);

	std::cout << Ann::round(Model.predict(x), 0.5).values();
	std::cout << "Accuracy: "<< 
		rx::Utility::accuracy(labels.values(), Ann::round(Model.predict(x), 0.5).T().values()) <<"%";

	return 0;

}