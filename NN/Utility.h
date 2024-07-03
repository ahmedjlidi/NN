#pragma once
#include "DataSet.h"

#define _CRT_SECURE_NO_WARNINGS

#define PRINT(X) std::cout<<"Process: "<<X<<"%\n";

namespace rx
{

	#define E 2.71828

	class Utility
	{
	public:
		static std::string cleanStr(std::string str);
		static bool isInt(const std::string& str);
		static bool isFloat(const std::string& str);

		template<typename T>
		T findMostFreq(std::vector<T>& v)
		{
			T mostfreq; int max = 0;
			for (const auto& e : v)
			{
				int curr = 0;
				for (const auto& k : v)
				{
					if (e == k)
						curr++;

				}
				if (curr > max)
				{
					max = curr;
					mostfreq = e;
				}
			}
			return mostfreq;
		}





		template<typename T>
		bool findInVec(const std::vector<T>& v, T key)
		{
			for (const auto& e : v)
			{
				if (e == key)
					return true;
			}
			return false;
		}

		static int randInt(int x, int y);
		static float randFloat(float x, float y);

		template<typename Algorithm>
		static float Test(Algorithm* algorithm, bool debug = false)
		{
			float oldk =  algorithm->getK();
			const float lim = std::floor((algorithm->getSet().getRowNum() * (100 - algorithm->getPercentage())) / static_cast<float>(100.f));
			SET& set = *algorithm->getSet().getSet();

			FILE* output = fopen("outTemp.txt", "w+");
			FILE* predicted = fopen("preTemp.txt", "w+");

			std::string t;
			algorithm = new Algorithm(algorithm->getSet(), 100);
			algorithm->setK(oldk);
			const float uni = 100.f / (((set.size()) - (algorithm->getSet().getRowNum() - lim))) / 2.f ; float curr = 0;

			for (int i = algorithm->getSet().getRowNum() - lim ; i < set.size(); i++)
			{
					curr += uni;
					std::vector<std::string> temp;
					t= set[i].second;
					fprintf(output, "%s\n", cleanStr(t).c_str());
					for (const auto& k : set[i].first)
					{
						temp.push_back(k);
					}
					t = algorithm->predict(temp);
					fprintf(predicted, "%s\n", cleanStr(t).c_str());
					if(debug)
						PRINT(curr);
			}

			fclose(output); fclose(predicted);
			float score = 100;
			float unit = 100 / static_cast<float>(lim);

			std::ifstream f1("outTemp.txt");
			std::ifstream f2("preTemp.txt");
			if (!f1.is_open() || !f2.is_open()) 
			{
				exit(EXIT_FAILURE);
			}
			int l1, l2;

			while (f1 >> l1 && f2 >> l2)
			{
				curr += uni;
				std::string x = std::to_string(l1), y = std::to_string(l2);
				

				if (cleanStr(x) != cleanStr(y))
					score -= unit;

				if (debug)
					PRINT(curr);
			}

			f1.close();
			f2.close();
			remove("outTemp.txt");
			remove("preTemp.txt");
			return score;
		}


		template<typename T>
		static void printVec(std::vector<T>&& v)
		{
			std::cout << "\n";
			for (const auto& e : v)
			{
				std::cout << e << " ";
			}
			std::cout << "\n";
		}

		template<typename T>
		static void printVec(std::vector<T>& v)
		{
			std::cout << "\n";
			for (const auto& e : v)
			{
				std::cout << e << " ";
			}
			std::cout << "\n";
		}

		static std::vector<float> strVecToFloat(std::vector<std::string>&& v)
		{
			std::vector<float> temp;
			for (const auto& e : v)
			{
				temp.push_back(atof(e.c_str()));
			}
			return temp;
		}

		static std::vector<float> strVecToFloat(std::vector<std::string>& v)
		{
			std::vector<float> temp;
			for (const auto& e : v)
			{
				temp.push_back(atof(e.c_str()));
			}
			return temp;
		}


		static std::vector<std::pair<float, float>> normalized(rx::SET& set);

		static float normalize(float x, float min, float max);

		static float dot(std::vector<float>& v1, std::vector<float>& v2);

		static float ReLU(float x)
		{
			return std::max(x, float(0));
		}
		static float Sigmoid(float x)
		{
			return 1.f / (1.f + std::exp(-x));
		}

		static float mean(std::vector<float>& v);

		static float loss(float y, float yHat);
		static float cost(std::vector<float>& losses);


		static float Bce(std::vector<float>& y, std::vector<float>& yHat);
		static std::vector<float> computeError(std::vector<float>& y, std::vector<float>& yHat);
		static float sum(std::vector<float>& v);

		static float accuracy(std::vector<std::vector<float>>& y, std::vector<std::vector<float>>& yHat);
		
		
	};
}
