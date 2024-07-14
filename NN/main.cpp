#include "stdafx.h"
#include "Ann.h"
#include "DataSet.h"

void setWeights(Ann& Model)
{
	std::vector<std::vector<float>> v1 = { {0.1, 0.1}, {0.1, 0.1} };
	std::vector<std::vector<float>> v2 = { {0.1, 0.1} };
	Model.setWeights(0, Tensor(v1));
	Model.setWeights(1, Tensor(v2));
}

int main()
{

	rx::DataSet dataset;
	if (!dataset.loadCsvFile("xor.csv"))
		return 1;

	std::pair<Tensor, Tensor> data = dataset.get_As_Tensor();
	Tensor x = data.first;
	Tensor y = data.second;

	Ann Model = Ann();
	Model.addLayer(2, 8, False);
	Model.addLayer(8, 1, True);

	//setWeights(Model);
	Ann::passData(x, y, Model);
	Model.compile(0.1, "ReLU", "Sigmoid");

	Model.train(400, True);


	////Ann::summary(Model);
	//
	////Ann::describe(Model);

	//print(Model.predict(dataset.get_As_Tensor().first).values());

	std::cout << Ann::round(Model.predict(x), 0.5).values();
	
	
	return 0;
}

