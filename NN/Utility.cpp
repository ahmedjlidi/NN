#include "stdafx.h"
#include "Utility.h"
using namespace rx;

std::string Utility::cleanStr(std::string str)
{
	if (str[str.size() - 1] == '\n')
		str = str.substr(0, str.size() - 1);
	return std::string(str);
}

bool Utility::isInt(const std::string& str)
{
	for (const auto& e : str)
		if (e <= '0' || e >= '9')
			return false;
	return true;

}

bool Utility::isFloat(const std::string& str)
{
	for (const auto& e : str)
	{
		if (!isdigit(e))
			return false;
		if (e == '.')
			continue;
	}
	return true;
}

int rx::Utility::randInt(int x, int y)
{
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist6(x, y);
	return dist6(rng);
}

float rx::Utility::randFloat(float x, float y)
{
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> dist(x, y);
	return dist(rng);
}

std::vector<std::pair<float, float>> rx::Utility::normalized(rx::SET& set)
{
	int count = 0;
	std::vector<std::pair<float, float>>temp;

	while (count < set[0].first.size())
	{
		int min = 999999, max = -999999;
		for (const auto& e : set)
		{
			int localCount = 0;
			for (const auto& row : e.first)
			{
				if (localCount == count)
				{
					if (atof(row.c_str()) > max)
						max = atof(row.c_str());
					if (atof(row.c_str()) < min)
						min = atof(row.c_str());
				}
				localCount++;
				
			}	
		}
		temp.push_back(std::make_pair(min, max));
		count++;
	}
	return temp;

}

float rx::Utility::normalize(float x, float min, float max)
{
	float v1 = x - min;
	float v2 = max - min;
	return static_cast<float>(v1) / v2;
}

float rx::Utility::dot(std::vector<float>& v1, std::vector<float>& v2)
{
	float total = 0.f;
	if (v1.size() != v2.size())
	{
		std::cerr << "V1 and V2 cannot be multpilied" << "v1 = " << v1.size() << " v2 = " << v2.size() << "\n";
		exit(1);
	}
	for (int i = 0; i < v1.size(); i++)
	{
		total += (v1[i] * v2[i]);
	}
	return total;
}

float rx::Utility::mean(std::vector<float>& v)
{
	float total = 0.f;
	for (const auto& e : v) 
	{
		total += e;
	}
	return total / static_cast<float>(v.size());
}

float rx::Utility::loss(float y, float yHat)
{
	return (y * std::log(yHat) + (1 - y) * std::log(1 - yHat)) * -1.f;
}

float rx::Utility::cost(std::vector<float>& losses)
{
	return rx::Utility::mean(losses);
}

float rx::Utility::Bce(std::vector<float>& y, std::vector<float>& yHat)
{
	float total = 0.f;
	auto bce = [](float y, float yHat) 
		{
		float z = y * std::log(yHat) + (1 - y) * std::log(1 - yHat);
		return z;
		};
	try
	{
		if (y.size() != yHat.size())
			throw std::runtime_error("input and output are not the same size.\n");

	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
	for (int i = 0; i < y.size(); i++)
	{
		total += bce(y[i], yHat[i]);
	}

	return (static_cast<float>(-1.f) / y.size()) * total;
}

std::vector<float> rx::Utility::computeError(std::vector<float>& y, std::vector<float>& yHat)
{
	try
	{
		if (y.size() != yHat.size())
			throw std::runtime_error("input and output are not the same size.\n");

	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
	std::vector<float> v;
	for (int i = 0; i < y.size(); i++)
	{
		v.push_back(yHat[i] - y[i]);
	}
	return v;
}

float rx::Utility::sum(std::vector<float>& v)
{
	float total = 0.f;
	for (const auto& e : v)
	{
		total += e;

	}
	return total;
}

float rx::Utility::accuracy(std::vector<std::vector<float>>* y, std::vector<std::vector<float>>* yHat)
{
	try
	{
		if (y->size() != yHat->size() || y[0].size() != yHat[0].size())
		{
			throw std::runtime_error("output and predicted value are not the same size. rx::Utility::accuracy.\n");

		}
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
	float total = y->size() * y[0].size();
	float curr = total;
	float offset = 100.f / static_cast<float>(total);
	for (int i = 0; i < y->size(); i++)
	{
		for (int j = 0; j < y[0].size(); j++)
		{
			if (static_cast<float>((*y)[i][j]) != (*yHat)[i][j])
				curr -= 1;
		}
	}
	return curr * offset;
}

float rx::Utility::accuracy(std::vector<std::vector<float>> y, std::vector<std::vector<float>> yHat)
{
	try
	{
		if (y.size() != yHat.size() || y[0].size() != yHat[0].size())
		{
			throw std::runtime_error("output and predicted value are not the same size. rx::Utility::accuracy.\n");

		}
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
		exit(1);
	}
	float total = y.size() * y[0].size();
	float curr = total;
	float offset = 100.f / static_cast<float>(total);
	for (int i = 0; i < y.size(); i++)
	{
		for (int j = 0; j < y[0].size(); j++)
		{
			if (static_cast<float>(y[i][j]) != yHat[i][j])
				curr -= 1;
		}
	}
	return curr * offset;
}

float rx::Utility::kaiming_init(int in)
{
	float n = static_cast<float>(in);

	std::random_device rd;
	std::mt19937 gen(rd());
	float start = -(std::sqrt(6.f / n)), end = start * -1.f;
	std::uniform_real_distribution<> dis(start, end);

	

	return dis(gen);
}

void rx::Utility::normalize(std::vector<std::vector<float>>& v)
{
	int count = 0;
	while (count < v[0].size())
	{
		
		float min = rx::Utility::min(v, count), max = rx::Utility::max(v, count);
		
		for (int i = 0; i < v.size(); i++)
		{;
			for (int j = 0; j <= count; j++)
			{
				if (j == count)
				{
					v[i][j] = ((v[i][j] - min) / ((static_cast<float>(max) - min) * 1.f));
				}
					
			}
		}
		count++;
	}
	
}

float rx::Utility::min(std::vector<std::vector<float>>& v, int ax)
{
	int min = 99999;
	int count = 0;
	for (const auto& e : v)
	{
		for (const auto& k : e)
		{
			if (count == ax)
			{
				if (k < min)
				{
					min = k;
				}
			}
			count++;
		}
		count = 0;
	}
	return min;
}

float rx::Utility::max(std::vector<std::vector<float>>& v, int ax)
{
	int max = -99999;
	int count = 0;
	for (const auto& e : v)
	{
		for (const auto& k : e)
		{
			if (count == ax)
			{
				if (k > max)
				{
					max = k;
				}
			}
			count++;
		}
		count = 0;
	}
	return max;
}




	