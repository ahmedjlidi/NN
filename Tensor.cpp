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



bool Tensor::empty()
{
	return this->tensor.mat.size() == 0;
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
		if (t.values().size() == 0 ||  this->values().size() == 0)
		{
			throw std::runtime_error("mat 1 and mat 2 cannot be added because one at least has size 0.\n");
		}
		if (t.values().size() != this->values().size())
		{
			throw std::runtime_error("mat1 and mat2 cannot be added.Not the same size.\n(" 
				+ std::to_string(this->tensor.mat.size())+ " x " + std::to_string(this->tensor.mat[0].size()) + ") and (" +
				std::to_string(t.tensor.mat.size()) + " x " + std::to_string(t.tensor.mat[0].size()));
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
		if (this->getShape().first == 0 || this->getShape().second == 0 || t.getShape().first == 0
			|| t.getShape().second == 0)
		{
			std::string n = (this->getShape().first == 0 || this->getShape().second == 0) ? "mat 1 " : "mat2 ";
			throw std::runtime_error(n + "Has 0 or rows or columns." + std::to_string(this->tensor.getRowNum()) + "x"
				+ std::to_string(this->tensor.getColNum()) + ") and (" + std::to_string(t.tensor.getRowNum()) + "x" +
				std::to_string(t.tensor.getColNum()) + ")");
		}

		if (this->tensor.getColNum() == t.tensor.getColNum() && this->tensor.getRowNum() == t.tensor.getRowNum())
		{
			Tensor temp;
			for (int z = 0; z < t.getShape().first; z++)
			{
				temp.values().push_back(std::vector<float>());
				for (int y = 0; y < t.getShape().second; y++)
				{
					temp.values()[z].push_back(this->tensor.mat[z][y] * t.values()[z][y]);
				}
			}

			return temp;
		}
		if (this->tensor.getColNum() != t.tensor.getRowNum())
		{
			std::cout << "--------------------------------------------\n";
			std::cout << this->tensor.mat<<"-------\n" << t.tensor.mat << "\n";
			std::cout << "--------------------------------------------\n";
			throw std::runtime_error("mat1 and mat2 cannot me multiplied.\n(" + std::to_string(this->tensor.getRowNum()) +"x" 
				+ std::to_string(this->tensor.getColNum()) + ") and (" + std::to_string(t.tensor.getRowNum()) + "x" +
				std::to_string(t.tensor.getColNum()) +")");
		}
		
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
	

	const auto& A = this->tensor.mat;
	const auto& B = t.tensor.mat;

	int rowsA = A.size();
	int colsA = A[0].size();
	int colsB = B[0].size();

	std::vector<std::vector<float>> C(rowsA, std::vector<float>(colsB, 0.0f));
	float sum = 0.f;
#pragma omp parallel for
	for (short i = 0; i < rowsA; ++i) 
	{
		for (short j = 0; j < colsB; ++j)
		{
			for (short k = 0; k < colsA; ++k)
			{
				sum += A[i][k] * B[k][j];
			}
			C[i][j] = sum;
			sum = 0.f;
		}
	}

	return Tensor(C);
}

Tensor Tensor::operator*(float scalar)
{

	auto scale = [](float& element, const float& scalar) -> void
		{
		element *= scalar;
		};
	auto processInVec = [scale](std::vector<float>& vec, float constant)
		{
		std::for_each(vec.begin(), vec.end(), [constant, scale](float& element)
			{
			scale(element, constant);
			});
		};
	std::for_each(this->tensor.mat.begin(), this->tensor.mat.end(), [scalar, processInVec](std::vector<float>& innerVec) {
		processInVec(innerVec, scalar);
		});

	return *this;
}

Tensor Tensor::operator+(float v)
{
	auto add = [](float& element, const float& scalar) -> void
		{
			element += scalar;
		};
	auto processInVec = [add](std::vector<float>& vec, float constant)
		{
			std::for_each(vec.begin(), vec.end(), [constant, add](float& element)
				{
					add(element, constant);
				});
		};
	std::for_each(this->tensor.mat.begin(), this->tensor.mat.end(), [v, processInVec](std::vector<float>& innerVec) {
		processInVec(innerVec, v);
		});

	return *this;
}

Tensor Tensor::operator-(Tensor& t)
{
	try
	{
		if (t.values().size() != this->values().size())
		{
			throw std::runtime_error("Cannot operate -. mat1 and mat2 are not the same size.\n(" +
				std::to_string(t.getShape().first) + 'x' + std::to_string(t.getShape().first) + ')' + " and " +

				'('+ std::to_string(this->values().size()) + 'x' + std::to_string(this->values()[0].size()) +')');
		}

		Tensor temp;
		for (int i = 0; i < t.values().size(); i++)
		{
			std::vector<float> vec;
			for (int j = 0; j < t.values()[0].size(); j++)
			{
				vec.push_back(this->values()[i][j] - t.values()[i][j]);
			}
			temp.values().push_back(vec);
		}
		return temp;
	}

	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
}

Tensor Tensor::operator-(float v)
{
	for (auto& e : this->tensor.mat)
	{
		for (auto& k : e)
		{
			k -= v;
		}
	}
	return *this;
}

Tensor Tensor::operator=(std::vector<float> v )
{
	this->tensor.mat.push_back(v);
	return *this;
}

Tensor Tensor::operator=(float v)
{
	this->tensor.mat.resize(1);
	this->tensor.mat[0].push_back(v);
	return *this;
}

