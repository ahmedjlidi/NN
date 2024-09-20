#include "stdafx.h"
#include "utils.h"

void train(Net *Model, int num_epochs, bool debug = true)
{
	int curr_epoch = 0;
	const int total_epochs = num_epochs;
	Timer *timer = nullptr;
	float total_time = 0.f;
	if (debug)
	{
		timer = new Timer();
		timer->start();
	}
	while (num_epochs--)
	{
		for (int i = 0; i < Model->getIterationSize(); i++)
		{
			Model->forward();
			Model->zero_grad();
			Model->backProp();
			Model->optimizer_step();
		}
		if (debug)
		{
			printf("Epoch %d/%d------Loss: %.3f   Time: %.2f\n",
				   curr_epoch, total_epochs, Model->getCurrLoss(), timer->elapsed_time());
			std::cout << std::flush;
			total_time += timer->elapsed_time();
			timer->reset();
		}
		curr_epoch++;
	}

	if (debug)
		printf("Total training time: %.2f\n", total_time);
}

int main(int argc, char *argv[])
{

	rx::DataSet dataset;
	std::string fName;
	if (argc == 2)
		fName = argv[1];
	else
	{
		std::cerr << "File name is required as an argument.\n";
		exit(1);
	}
	if (!dataset.loadCsvFile(path(fName)))
	{

		std::cerr << "File was not found.\n " + path(fName);
		return 1;
	}

	printNode();

	printf("\nDataset shape: ");
	dataset.info();

	Tensor y;
	y.values() = rx::Utility::labelEncode(dataset.getSet());
	y = y.T();
	std::pair<Tensor, Tensor> data = dataset.get_As_Tensor();
	Tensor x = data.first;
	rx::Utility::normalize(x.values());

	auto Model = rx::initNet(getModelType());

	std::pair<float, int> lr_epochs = constructModel(*Model);
	Net::passData(x, y, *Model);
	Model->setLR(lr_epochs.first);
	train(Model.get(), 1000, True);
	Tensor output;
	output.values().resize(1);
	// std::cout << "Model accuracy: " << rx::Utility::accuracy(y.values(), Net::round(Model->predict(x), 0.5).T().values());

	for (int i = 0; i < Model->getInput().values().size(); i++)
	{
		output.values()[0].push_back(rx::Utility::argmax(Model->predict(x).values()[i]));
	}
	printf("Model accuracy: %.2f %c \n", rx::Utility::accuracy(y.values(), output.values()), '%');
	return 0;
}