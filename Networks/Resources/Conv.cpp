#include "../../stdafx.h"
#include "Conv.h"

Conv2d::Conv2d(int in_channel, int out_channel, int kernel_size, int stride, int padding)
{
    kernel = new Kernel(kernel_size, in_channel, 0.1);

    this->parameters["in_channel"] = in_channel;
    this->parameters["out_channel"] = out_channel;
    this->parameters["kernel_size"] = kernel_size;
    this->parameters["stride"] = stride;
    this->parameters["padding"] = padding;
}

Conv2d::~Conv2d()
{
    delete kernel;
}

void Conv2d::forward(Tensor &x)
{
    try
    {
        if (x.getShape().first < 1 || x.getShape().second < 1)
            throw std::runtime_error("Tensor has invalid size");
        else if (x.getShape().first < this->kernel->size || x.getShape().second < this->kernel->size)
        {
            throw std::runtime_error("Tensor size should be equal or greater than kernel size");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
    int cur_start_x, curr_start_y = 0;
    float total = 0;
    std::vector<float> sub_vec;
    for (int i = 0; i < curr_start_y; i++)
    {
        for (int j = cur_start_x; j < x.values().size(); j++)
        {
            for (int k = j; k < this->parameters["kernel_size"]; k++)
            {
                for (int h = i; h < this->parameters["kernel_size"]; h++)
                {
                    sub_vec.push_back(x.values()[])
                }
            }
        }
    }
    for (int i = 0; i < this->parameters["kernel_size"] || i < x.values()[0].size(); i++)
    {
        sub_vec.push_back(x.values()[curr_x][i]);
        curr_y++;
    }
    printf("Conv2d sum = %.2f", total);
}
