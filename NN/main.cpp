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

	/*rx::DataSet dataset;
	if (!dataset.loadCsvFile("xor.csv"))
		return 1;*/

	/*std::pair<Tensor, Tensor> data = dataset.get_As_Tensor();
	Tensor x = data.first;
	Tensor y = data.second;*/
	std::vector<std::vector<float>> x = { {1,1}};
	std::vector<float> y = { 0 };


	Ann Model = Ann();
	Model.addLayer(2, 2, True);
	Model.addLayer(2, 1, True);

	setWeights(Model);
	Ann::passData(x, y, Model);
	Model.compile(0.1, "ReLU", "Sigmoid");

	Model.train(1);
	


	////Ann::summary(Model);
	//
	////Ann::describe(Model);

	//print(Model.predict(dataset.get_As_Tensor().first).values());

	std::cout << Ann::round(Model.predict(x), 0.5).values();
	
	
	return 0;
}

