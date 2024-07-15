#include "stdafx.h"
#include "Ann.h"
#include "DataSet.h"

void setWeights(Ann& Model)
{
	std::vector<std::vector<float>> v1 = { {0.1, 0.1}, {0.1, 0.1} }, v2 = v1;
	std::vector<std::vector<float>> v3 = { {0.1, 0.1} };
	Model.setWeights(0, Tensor(v1));
	Model.setWeights(1, Tensor(v2));
	Model.setWeights(2, Tensor(v3));
}

int main()
{

	/*rx::DataSet dataset;
	if (!dataset.loadCsvFile("xor.csv"))
		return 1;

	std::pair<Tensor, Tensor> data = dataset.get_As_Tensor();
	Tensor x = data.first;
	Tensor y = data.second;*/

	std::vector<std::vector<float>> x = { {1,1}};
	std::vector<float> y = { 0 };


	Ann Model = Ann();
	Model.addLayer(2, 2, True);
	Model.addLayer(2, 2, True);
	Model.addLayer(2, 1, True);

	setWeights(Model);
	Ann::passData(Tensor(x), Tensor(y), Model);
	Model.compile(0.1, "ReLU", "Sigmoid");
	//Model.forward();
	
	Model.train(1);
	Model.debug();

	////Ann::summary(Model);
	//
	////Ann::describe(Model);
	//print(Model.predict(x).values());

	//std::cout << Ann::round(Model.predict(x), 0.5).values();
	
	return 0;
}

