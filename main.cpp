#include "stdafx.h"
#include "Ann.h"
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
	const char* homeDir = getenv("HOME");
    if (homeDir == nullptr) {
        std::cerr << "Could not retrieve home directory.\n";
        exit(1);
    }

    return std::string(homeDir) + "/src/CSVs/" + fName;
}

void printNode()
{
	std::FILE* f = fopen(path("note.txt").c_str(), "r");
	if(!f)
	{	printf("file not fdound.\n");
		return;
	}
	char c = fgetc(f);
	while(!feof(f))
	{
		printf("%c", c);
		c = fgetc(f);
	}
	fclose(f);
}

std::pair<float, int> constructModel(Ann& Model)
{
	printf("Add layers to the network (enter -1 to finish).\n");
	std::pair<int, int> options;
	printf("Enter input and output to the %d layer: ", (int)Model.getLayers().size());
	std::cin>> options.first >> options.second;
	while(options.first != -1 && options.second != -1)
	{
		Model.addLayer(options.first, options.second, True);
		printf("Enter input and output to the %d layer: ", (int)Model.getLayers().size());
		std::cin>> options.first >> options.second;
	}
	
	std::pair<float, int> lr_epochs;
	printf("Enter learning rate and epochs: ");
	std::cin>>lr_epochs.first>> lr_epochs.second;

	return lr_epochs;
}

void display_menu(Ann& Model, int epochs, Tensor& x, Tensor& y) {
    int option = 0;
    std::cout << "Options:\n1) Review Model\t\t2) Start Training\t\t3) Exit\n";
    std::cin >> option;

    while (option != 3) {
        switch (option) {
            case 1:
                Ann::info(Model);
                Ann::summary(Model);
                break;
            case 2:
                {
                    float time = 0.f;
                    Timer timer;
                    timer.start();
                    Model.train(epochs, true, false); // Assuming True and False are meant to be true and false
                    time = timer.elapsed_time();
                    std::cout << "Model accuracy: " << rx::Utility::accuracy(y.values(), Ann::round(Model.predict(x), 0.5).T().values()) 
                              << "\nTraining time: " << time << "s\n";
                }
                break;    
            case 3:
                exit(0);
                break;
            default:
                std::cout << "Invalid option.\n";
                break;
        }
        std::cout << "Options:\n1) Review Model\t\t2) Start Training\t\t3) Exit\n";
        std::cin >> option;
    }
}


int main(int argc, char* argv[])
{

	rx::DataSet dataset;
	std::string fName;
	if(argc == 2)
	    fName = argv[1];
	else{
	    std::cerr<<"File name is required as an argument.\n";
	    exit(1);	
	}
	if (!dataset.loadCsvFile(path(fName))){
		std::cerr<<"File was not found.\n";
		return 1;
	}

	printNode();

	printf("\nDataset shape: ");
	dataset.info();

		
	std::pair<Tensor, Tensor> data = dataset.get_As_Tensor();
	Tensor x = data.first;
	Tensor y = data.second;

	rx::Utility::normalize(x.values());
	Ann Model = Ann();


	std::pair<float, int> lr_epochs = constructModel(Model);
	Ann::passData(x, y, Model);
	Model.compile(lr_epochs.first, "ReLU", "Sigmoid");

	
	display_menu(Model, lr_epochs.second, x, y);




	return 0;
}

