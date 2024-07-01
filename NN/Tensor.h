#pragma once
#include "Operators.h"


class matrix
{

private:
	friend class Tensor;
	std::vector < std::vector<float>> mat;

	std::vector<float> getCol(int j)
	{
		std::vector<float> temp;
		for (int i = 0; i < mat.size(); i++)
		{
			temp.push_back(mat[i][j]);
		}

		return temp;
	}
	std::vector<float> getRow(int i)
	{
		return std::vector<float>(mat[i]);
	}

	int getRowNum()
	{
		return mat.size();
	}
	int getColNum()
	{
		return mat[0].size();
	}
public:
	
	
};



class Tensor
{
public:
	
	typedef std::vector < std::vector<float>> MATRIX;

private:
	 typedef matrix TENSOR;
	 TENSOR tensor; 

public:
	Tensor(std::vector<std::vector<float>> &v);
	Tensor(std::vector<float>& v);
	Tensor();

	static bool multiplicationValid(int row, int col);

	//Shape and description//////
	std::pair<int, int> getShape();
	////////////////////////////////


	//Return Values///////////////////////////////
	std::vector < std::vector<float>>& values();
	///////////////////////////////////////////


	std::vector<float> squeeze(bool horizontal = true);
	Tensor T();

	Tensor colAdd(std::vector<float>&vals);
	//Operations////////////////
	Tensor operator+(Tensor& t);
	Tensor operator*(Tensor& t);
	Tensor operator*(float scalar);
	Tensor operator+(float v);
	Tensor operator+(std::vector<float> v);
	Tensor operator-(Tensor& t);
	Tensor operator=(std::vector<float>& v);
	////////////////////////////

	

	//Operators///////////////////////

	//print a vector
	//friend std::ostream& operator<<(std::ostream& os, Tensor& t)
	/*{
		for (const auto& row : t.values())
		{
			os << "[ ";
			for (int j = 0; j < row.size(); j++)
			{
				os << row[j];
				if (j + 1 < row.size())
				{
					os << ", ";
				}
			}
			os << "]\n";
		}
		return os;
	}*/

	//print a matrix
	//friend std::ostream& operator<<(std::ostream& os, Tensor& t);

	/////////////////////////////////
};






