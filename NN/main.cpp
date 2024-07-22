#include "stdafx.h"
#include "Ann.h"
#include "DataSet.h"

void setWeights(Ann& Model)
{
	std::vector<std::vector<float>> v1 = { {0.1, 0.1}, {0.1, 0.1} }, v2 = v1, v3 = v1;
	//std::vector<std::vector<float>> v2 = { {0.1, 0.1}, };
	std::vector<std::vector<float>>v4 = { {0.1}, {0.1} };
	Model.setWeights(0, Tensor(v1));
	Model.setWeights(1, Tensor(v2));
	Model.setWeights(2, Tensor(v4));
	//Model.setWeights(3, Tensor(v4));
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
	Model.addLayer(2, 16, True);
	Model.addLayer(16, 1, True);


	Ann::passData(x, y, Model);
	Model.compile(0.1, "ReLU", "Sigmoid");

	Model.train(30, True, True);

	Tensor output = y;
 	printf("Model accuracy: %.2f \% \n", rx::Utility::accuracy(output.values(), 
		 Ann::round(Model.predict(x), 0.5).T().values()));
	print(Model.predict(x).values());

	
	return 0;
}

