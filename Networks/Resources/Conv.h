#ifndef CONV_H
#define CONV_H

#include "Layer.h"

struct Kernel
{
    int size;
    Tensor mat;
    Kernel(int size, int input_size = 1, float constant = -200.5) : size(size)
    {
        if (size % 2 == 0)
        {
            printf("Kernel size should be odd. new kernel size %d", size - 1);
        }
        for (int i = 0; i < size; i++)
        {
            mat.values().push_back(std::vector<float>());
            for (int j = 0; j < size; j++)
            {
                if (constant == 200.5)
                    mat.values()[i].push_back(rx::Utility::kaiming_init(input_size));
                else
                    mat.values()[i].push_back(constant);
            }
        }
    }
};

class Conv2d
{
private:
    int kernel_size;
    int padding;
    int stride;
    int in_channel, out_channel;
    Kernel *kernel;

public:
    Conv2d(int in_channel, int out_channel, int kernel_size, int stride, int padding);
    virtual ~Conv2d();

    std::map<std::string, int> parameters;
    void forward(Tensor &x);
};

#endif