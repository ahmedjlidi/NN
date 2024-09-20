#include "Networks/Network.h"
#include "DataSet.h"

std::string path(std::string fName);
void printNode();
Tensor getInput(int size);

template <typename NN>
std::pair<float, int> constructModel(NN &Model);

template <typename NN>
void display_menu(NN &Model, int epochs, Tensor &x, Tensor &y);

int getModelType();

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

int getModelType()
{
    short option = -1;

    printf("Specify Model Type:\n1)Binary classication\t\t2)Cross classification\t\t"
           "3)Regression\nEnter: ");
    std::cin >> option;
    while (option < 1 || option > 3)
    {
        printf("Specify Model Type:\n1)Binary classication\t\t2)Cross classification\t\t"
               "3)Regression\nEnter: ");
        std::cin >> option;
    }
    return option;
}

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
            Model.train(epochs, True);
            time = timer.elapsed_time();

            if (Model.getLayers()[Model.getLayers().size() - 1]->features().second == 1)
            {
                std::cout << "Model accuracy: " << rx::Utility::accuracy(y.values(), Net::round(Model.predict(x), 0.5).T().values())
                          << "\nTraining time: " << time << "s\n";
            }
            else if (Model.getLayers()[Model.getLayers().size() - 1]->features().second > 1)
            {
                std::cout << "Training time: " << time << "s\n";
                Tensor output;
                output.values().resize(1);
                for (int i = 0; i < Model.getInput().values().size(); i++)
                {
                    output.values()[0].push_back(rx::Utility::argmax(Model.predict(x).values()[i]));
                }
                printf("Model accuracy: %.2f %c \n", rx::Utility::accuracy(y.values(), output.values()), '%');
            }
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