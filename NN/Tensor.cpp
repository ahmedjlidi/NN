#include "stdafx.h"
#include "Tensor.h"



Tensor::Tensor(std::vector<std::vector<float>>& v)
{
	this->tensor.mat = v;
}
Tensor::Tensor(std::vector<float>& v)
{
	this->tensor.mat.push_back(v);
}
Tensor::Tensor()
{
}

bool Tensor::multiplicationValid(int row, int col)
{
	return row == col;
}

std::pair<int, int> Tensor::getShape()
{
	return std::make_pair(this->tensor.getRowNum(), this->tensor.getColNum());
}



std::vector<float> Tensor::squeeze(bool horizontal)
{
	if(horizontal)
		return this->tensor.mat[0];
	else
	{
		std::vector<float> v;
		for (auto& e : this->tensor.mat)
		{
			v.push_back(e[0]);
		}
		return v;
	}
}

Tensor Tensor::T()
{
	TENSOR* temp = new TENSOR();
	
	
	for (int i = 0; i < this->tensor.getColNum(); i++)
	{
		temp->mat.push_back(this->tensor.getCol(i));
	}
	Tensor x = Tensor(temp->mat);
	return x;
}

Tensor Tensor::colAdd(std::vector<float>& vals)
{
	try
	{
		if (vals.size() != this->tensor.getColNum())
		{
			throw std::runtime_error("Cannot scale mat by vec. different in sizes.\n" +std::to_string(vals.size()) +
			" and " + std::to_string(this->tensor.getColNum()));

		}

	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
	int count = 0;
	while(count < this->tensor.mat[0].size())
	{
		for (int i = 0; i < this->tensor.mat.size(); i++)
		{
			this->tensor.mat[i][count] += vals[count];
		}
		count++;
	}
	return *this;
}

std::vector < std::vector<float>>& Tensor::values()
{
	return this->tensor.mat;
}

Tensor Tensor::operator+(Tensor& t)
{
	try
	{
		if (t.values().size() != this->values().size())
		{
			throw std::runtime_error("mat1 and mat2 are not the same size.\n" + std::to_string(t.values().size()) + "and" +
				 std::to_string(this->values().size()));
		}
		Tensor* temp = new Tensor();
		for (int i = 0; i < t.values().size(); i++)
		{
			std::vector<float> vec; 
			for (int j = 0; j < t.values()[0].size(); j++)
			{
				vec.push_back(t.values()[i][j] + this->values()[i][j]);
			}
			temp->values().push_back(vec);
		}
		return *temp;
	}

	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}

}

Tensor Tensor::operator*(Tensor& t)
{
	try
	{
		if (this->tensor.getColNum() != t.tensor.getRowNum())
		{
			std::cout << this->tensor.getColNum() << " " << t.tensor.getRowNum() << "\n";
			throw std::runtime_error("mat1 and mat2 are not the same size.\n(" + std::to_string(t.tensor.getRowNum()) +"x" 
				+ std::to_string(t.tensor.getColNum()) + ") and (" + std::to_string(t.tensor.getRowNum()) + "x" +
				std::to_string(t.tensor.getColNum()) +")");
		}
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
	int rowsA = this->tensor.mat.size();
	int colsA = this->tensor.mat[0].size();
	int colsB = t.tensor.mat[0].size();

	std::vector<std::vector<float>> C(rowsA, std::vector<float>(colsB, 0.0f));
	for (int i = 0; i < rowsA; ++i) 
	{
		for (int j = 0; j < colsB; ++j)
		{
			for (int k = 0; k < colsA; ++k) 
			{
				C[i][j] += this->tensor.mat[i][k] * t.tensor.mat[k][j];
			}
		}
	}
	Tensor* temp = new Tensor(C);
	return *temp;
}

Tensor Tensor::operator*(float scalar)
{
	for (auto& e : this->tensor.mat)
	{
		for (auto& k : e)
		{
			k *= scalar;
		}
	}
	return *this;
}

Tensor Tensor::operator+(float v)
{
	for (auto& e : this->tensor.mat)
	{
		for(auto& k: e)
			k += v;
	}
	return *this;
}

Tensor Tensor::operator-(Tensor& t)
{
	try
	{
		if (t.values().size() != this->values().size())
		{
			throw std::runtime_error("mat1 and mat2 are not the same size.\n" + std::to_string(t.values().size()) + "and" +
				std::to_string(this->values().size()));
		}

		Tensor* temp = new Tensor();
		for (int i = 0; i < t.values().size(); i++)
		{
			std::vector<float> vec;
			for (int j = 0; j < t.values()[0].size(); j++)
			{
				vec.push_back(t.values()[i][j] - this->values()[i][j]);
			}
			temp->values().push_back(vec);
		}
		return *temp;
	}

	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
}

Tensor Tensor::operator=(std::vector<float>&v )
{
	this->tensor.mat.push_back(v);
	return *this;
}

