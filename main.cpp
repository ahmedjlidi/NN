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

std::string path(std::string fName)
{
	const char *homeDir = getenv("HOME");
	if (homeDir == nullptr)
	{
		std::cerr << "Could not retrieve home directory.\n";
		exit(1);
	}

	return std::string(homeDir) + "/NN/CSVs/" + fName;
}

void printNode()
{
	std::FILE *f = fopen(path("note.txt").c_str(), "r");
	if (!f)
	{
		printf("file not fdound.\n");
		return;
	}
	char c = fgetc(f);
	while (!feof(f))
	{
		printf("%c", c);
		c = fgetc(f);
	}
	fclose(f);
}

Tensor getInput(int size);

template <typename NN>
std::pair<float, int> constructModel(NN &Model)
{
	printf("Add layers to the network (enter -1 to finish).\n");
	std::pair<int, int> options;
	printf("Enter input and output to the %d layer: ", (int)Model.getLayers().size());
	std::cin >> options.first >> options.second;
	while (options.first != -1 && options.second != -1)
	{
		Model.addLayer(options.first, options.second, True);
		printf("Enter input and output to the %d layer: ", (int)Model.getLayers().size());
		std::cin >> options.first >> options.second;
	}

	std::pair<float, int> lr_epochs;
	printf("Enter learning rate and epochs: ");
	std::cin >> lr_epochs.first >> lr_epochs.second;

	return lr_epochs;
}

template <typename NN>
void display_menu(NN &Model, int epochs, Tensor &x, Tensor &y)
{
	Tensor test_input;
	int option = 0;
	std::cout << "Options:\n1) Review Model\t\t2) Start Training\t\t3) Test input\t\t4)Exit\n";
	std::cin >> option;
	while (option != 4)
	{
		switch (option)
		{
		case 1:
			NN::info(Model);
			NN::summary(Model);
			break;
		case 2:
		{
			float time = 0.f;
			Timer timer;
			timer.start();
			Model.train(epochs, true, True); // Assuming True and False are meant to be true and false
			time = timer.elapsed_time();
			std::cout << "Model accuracy: " << rx::Utility::accuracy(y.values(), NN::round(Model.predict(x), 0.5).T().values())
					  << "\nTraining time: " << time << "s\n";
		}
		break;
		case 3:
			std::cout << "Testing.\n";
			test_input = getInput(Model.getLayers()[0]->features().first);
			print(Model.predict(test_input).values());
			break;
		case 4:
			exit(0);
			break;
		default:
			std::cout << "Invalid option.\n";
			break;
		}
		std::cout << "Options:\n1) Review Model\t\t2) Start Training\t\t3) Test input\t\t4)Exit\n";
		std::cin >> option;
	}
}

template <typename NN>
void setWeights(NN &Model)
{
	std::vector<std::vector<float>> v1 = {{0.1, 0.1}, {0.1, 0.1}};
	std::vector<std::vector<float>> v2 = {{0.1, 0.1}};
	Model.setWeights(0, v1);
	Model.setWeights(1, v2);
}

int main()
{
	rx::DataSet dataset;
	if (!dataset.loadCsvFile(path("Iris.csv")))
	{
		std::cout << "File does not exist.\n" + path("mnist.csv");
		return EXIT_FAILURE;
	}
	Tensor y;
	y.values() = rx::Utility::labelEncode(dataset.getSet());
	y = y.T();
	std::pair<Tensor, Tensor> data = dataset.get_As_Tensor();
	Tensor x(data.first); // y(data.second); // y(data.second);
	rx::Utility::normalize(x.values());

	auto Model = rx::initNet(rx::TYPE::CROSS_CLASSIFICATION);
	Model->addLayer(5, 64);
	Model->addLayer(64, 64);
	Model->addLayer(64, 3);

	Model->passData(x, y, *Model);
	Model->setLR(0.02);

	float time = 0.f;
	Timer *timer = new Timer();
	timer->start();
	Model->train(1000, True);
	time = timer->elapsed_time();
	delete timer;

	std::cout << "Training time: " << time << "s\n";
	Tensor output;
	output.values().resize(1);
	for (int i = 0; i < dataset.getSet()->size(); i++)
	{
		output.values()[0].push_back(rx::Utility::argmax(Model->predict(x).values()[i]));
	}
	printf("Model accuracy: %.2f %c \n", rx::Utility::accuracy(y.values(), output.values()), '%');
	return 0;
}
// }

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

// 		std::cerr << "File was not found.\n "+path(fName) ;
// 		return 1;
// 	}

// 	printNode();

// 	printf("\nDataset shape: ");
// 	dataset.info();

// 	std::pair<Tensor, Tensor> data = dataset.get_As_Tensor();
// 	Tensor x = data.first;
// 	Tensor y = data.second;

// 	rx::Utility::normalize(x.values());

// 	auto Model = rx::initNet(rx::TYPE::BINARY_CLASSIFICATION);

// 	std::pair<float, int> lr_epochs = constructModel(*Model);
// 	Net::passData(x, y, *Model);
// 	Model->setLR(lr_epochs.first);
// 	display_menu<Net>(*Model, lr_epochs.second, x, y);
// 	return 0;
// }

Tensor getInput(int size)
{
	Tensor input;
	input.values().resize(1);
	std::cout << "Enter input: ";
	float value = 0.f;
	for (int i = 0; i < size; i++)
	{
		std::cin >> value;
		input.values()[0].push_back(value);
	}
	return input;
}
