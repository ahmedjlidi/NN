#include "stdafx.h"
#include "Ann.h"
#include "DataSet.h"



/*===================================================*/
/*Artifial neural network in c++ (FFN)
# Model only supports up to 2 layers currently
# Model uses relu activation function for hidden and sigmoid function for output layer
# Model only supports binary-classification currently (0 or 1)
# Model uses stochastic gradident decsent for backpropagation
# Model uses kaiming weight inisalization for weights (biases are set to 0)


@ Dataset class is used to read csv files
@ Dataset::get_as_tensor method returns 2 tensors. data and labels
@ Ann::passdata is used to pass data and labels for model
@ use model addlayer to add layers (up to 2)
@ ann::compile is used to set hyper-parameters (activation functions, learning_rate)
@Ann::info, summary, describe are used to describe the model
@ model::train trains the model, option True is set to print epochs, loss and accuracy


&{Feel free to improve the model}
====================================================*/

int main()
{

	rx::DataSet dataset;
	if (!dataset.loadCsvFile("xor.csv"))
		return 1;

	std::pair<Tensor, Tensor> data = dataset.get_As_Tensor();
	Tensor x = data.first;
	Tensor y = data.second;



	Ann Model = Ann();
	Model.addLayer(2, 8, True);
	Model.addLayer(8, 1, True);

	Ann::passData(Tensor(x), Tensor(y), Model);
	Model.compile(0.1, "ReLU", "Sigmoid");

	int option = -1;
	while (option != 1 || option != 2 || option != 3)
	{
		printf("1)Review Model\t\t2)Start training\t\t3)Exit\nEnter an option: ");
		scanf("%d", &option);
		switch (option)
		{
		case 1:
			Ann::info(Model);
			Ann::summary(Model);
			Ann::describe(Model);
			break;
		case 2:
			Model.train(1000, True);
			break;
		case 3:
			print("--------------------------");
			print("\nPredicted output:\n");
			print(Ann::round(Model.predict(x), 0.5).values(), 1);
			printf("Accuracy: %.3f % \n", (rx::Utility::accuracy(y.values(),Ann::round(Model.predict(x), 0.5).T().values())));
			print("\n--------------------------\n");
			break;
		default:
			printf("Invalid option.\nExiting.....\n");
			return 0;
		}
	}
	

	return 0;
}

