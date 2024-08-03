#include "stdafx.h"
#include "Networks/Network.h"
#include "DataSet.h"
/*
==========================================================================================
FFN v1.2.1
*Network only supports relu for hidden layers and sigmoid for ouptput layer
*Network only support binary classifcation (BCE loss is used)
*Adjust the model layers, learning rate and training epochs
*Network uses kaiming weight inisalizations and biasies are set to 0
*Network use batch-gradient-decsent to optimize solution
*Provide the csv dataset name as an argument. By default, they will be in CSVs directory
===========================================================================================
*/

// std::string path(std::string fName)
// {
// 	const char *homeDir = getenv("HOME");
// 	if (homeDir == nullptr)
// 	{
// 		std::cerr << "Could not retrieve home directory.\n";
// 		exit(1);
// 	}

// 	return std::string(homeDir) + "/src/CSVs/" + fName;
// }

// void printNode()
// {
// 	std::FILE *f = fopen(path("note.txt").c_str(), "r");
// 	if (!f)
// 	{
// 		printf("file not fdound.\n");
// 		return;
// 	}
// 	char c = fgetc(f);
// 	while (!feof(f))
// 	{
// 		printf("%c", c);
// 		c = fgetc(f);
// 	}
// 	fclose(f);
// }

// Tensor getInput(int size);

// template <typename NN>
// std::pair<float, int> constructModel(NN &Model)
// {
// 	printf("Add layers to the network (enter -1 to finish).\n");
// 	std::pair<int, int> options;
// 	printf("Enter input and output to the %d layer: ", (int)Model.getLayers().size());
// 	std::cin >> options.first >> options.second;
// 	while (options.first != -1 && options.second != -1)
// 	{
// 		Model.addLayer(options.first, options.second, True);
// 		printf("Enter input and output to the %d layer: ", (int)Model.getLayers().size());
// 		std::cin >> options.first >> options.second;
// 	}

// 	std::pair<float, int> lr_epochs;
// 	printf("Enter learning rate and epochs: ");
// 	std::cin >> lr_epochs.first >> lr_epochs.second;

// 	return lr_epochs;
// }

// template <typename NN>
// void display_menu(NN &Model, int epochs, Tensor &x, Tensor &y)
// {
// 	Tensor test_input;
// 	int option = 0;
// 	std::cout << "Options:\n1) Review Model\t\t2) Start Training\t\t3) Test input\t\t4)Exit\n";
// 	std::cin >> option;
// 	while (option != 4)
// 	{
// 		switch (option)
// 		{
// 		case 1:
// 			NN::info(Model);
// 			NN::summary(Model);
// 			break;
// 		case 2:
// 		{
// 			float time = 0.f;
// 			Timer timer;
// 			timer.start();
// 			Model.train(epochs, true, True); // Assuming True and False are meant to be true and false
// 			time = timer.elapsed_time();
// 			// std::cout << "Model accuracy: " << rx::Utility::accuracy(y.values(), NN::round(Model.predict(x), 0.5).T().values())
// 			// 		  << "\nTraining time: " << time << "s\n";

// 			std::cout << Model.predict(x).values() << y.values();
// 		}
// 		break;
// 		case 3:
// 			std::cout << "Testing.\n";
// 			test_input = getInput(Model.getLayers()[0]->features().first);
// 			print(Model.predict(test_input).values());
// 			break;
// 		case 4:
// 			exit(0);
// 			break;
// 		default:
// 			std::cout << "Invalid option.\n";
// 			break;
// 		}
// 		std::cout << "Options:\n1) Review Model\t\t2) Start Training\t\t3) Test input\t\t4)Exit\n";
// 		std::cin >> option;
// 	}
// }

// template <typename NN>
// void setWeights(NN &Model)
// {
// 	std::vector<std::vector<float>> v1 = {{0.1, 0.1}, {0.1, 0.1}};
// 	std::vector<std::vector<float>> v2 = {{0.1, 0.1}};
// 	Model.setWeights(0, v1);
// 	Model.setWeights(1, v2);
// }

// int main()
// {

// 	std::vector<std::vector<float>> input = {{1, 1}, {0, 0}, {1, 0}, {0, 1}};
// 	std::vector<std::vector<float>> output = {{0}, {0}, {1}, {1}};
// 	Tensor x(input), y(output);

// 	auto Model = Ann();
// 	Model.addLayer(2, 8, True);
// 	Model.addLayer(8, 1, True);
// 	Model.setWeights(0.1);
// 	Ann::passData(x, y, Model);
// 	Model.compile(0.1, "ReLU", "Sigmoid");
// 	Model.train(100, True);
// 	Model.describe(Model);

// 	auto Model2 = MAnn();
// 	Model2.addLayer(2, 8, True);
// 	Model2.addLayer(8, 1, True);
// 	Model2.setWeights(0.1);
// 	MAnn::passData(x, y, Model2);
// 	Model2.compile(0.1, "ReLU", "Sigmoid");
// 	Model2.train(100, True);
// 	Model2.describe(Model2);
//}
int main()
{

	rx::DataSet dataset;
	if (!dataset.loadCsvFile("/home/ahmed/src/CSVs/xor.csv"))
		return 1;

	std::pair<Tensor, Tensor> data = dataset.get_As_Tensor();
	Tensor x = data.first;
	Tensor y = data.second;

	rx::Utility::normalize(x.values());

	auto Model = rx::initNet(rx::TYPE::BINARY_CLASSIFICATION);
	Model->addLayer(2, 16, True);
	Model->addLayer(16, 1, True);

	std::cout << x.values() << y.values();

	Net::passData(x, y, *Model);
	Model->setLR(0.1);
	print("Train:\n");
	Model->train(1000, True, False);
	print(Model->predict(x).values());

	printf("Model accuracy: %.2f \n", rx::Utility::accuracy(y.values(),
															Net::round(Model->predict(x), 0.5).T().values()));

	return 0;
}
// int main(int argc, char *argv[])
// {

// 	rx::DataSet dataset;
// 	std::string fName;
// 	if (argc == 2)
// 		fName = argv[1];
// 	else
// 	{
// 		std::cerr << "File name is required as an argument.\n";
// 		exit(1);
// 	}
// 	if (!dataset.loadCsvFile(path(fName)))
// 	{
// 		std::cerr << "File was not found.\n";
// 		return 1;
// 	}

// 	printNode();

// 	printf("\nDataset shape: ");
// 	dataset.info();

// 	std::pair<Tensor, Tensor> data = dataset.get_As_Tensor();
// 	Tensor x = data.first;
// 	Tensor y = data.second;

// 	// rx::Utility::normalize(x.values());

// 	std::cout << x.values() << y.values();

// 	// std::vector<std::vector<float>> input = {{1, 1}, {0, 0}, {1, 0}, {0, 1}};
// 	// std::vector<float> output = {0, 0, 1, 1};
// 	// Tensor x(input), y(output);
// 	auto Model = Ann();

// 	// std::pair<float, int> lr_epochs = constructModel(Model);
// 	Model.addLayer(2, 16);
// 	Model.addLayer(16, 1);
// 	// Model.setWeights(0.1);
// 	Ann::passData(x, y, Model);
// 	Model.compile(0.1, "ReLU", "Sigmoid");
// 	Model.train(400, True);

// 	std::cout << "Model accuracy: " << rx::Utility::accuracy(y.values(), Ann::round(Model.predict(x), 0.5).T().values())
// 			  << "\nTraining time: " << time << "s\n";
// 	// display_menu<Ann>(Model, lr_epochs.second, x, y);
// 	return 0;
// }

// Tensor getInput(int size)
// {
// 	Tensor input;
// 	input.values().resize(1);
// 	std::cout << "Enter input: ";
// 	float value = 0.f;
// 	;
// 	for (int i = 0; i < size; i++)
// 	{
// 		std::cin >> value;
// 		input.values()[0].push_back(value);
// 	}
// 	return input;
// }
