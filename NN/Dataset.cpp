#include "stdafx.h"
#include "DataSet.h"


using namespace rx;

template<typename T>
 bool DataSet::findInVec(std::vector<T> v, T key)
{
	for (const auto& e : v)
	{
		if (e == key)
			return true;
	}
	return false;
}


 std::string DataSet::cleanStr(std::string STR)
{
	if (STR[STR.size() - 1] == '\n')
		STR = STR.substr(0, STR.size() - 1);
	return STR;
}


 bool DataSet::isInt(const std::string TEST)
{
	for (const auto& e : TEST)
		if (e <= '0' || e >= '9')
			return false;
	return true;

}

bool DataSet::isFLoat(const std::string TEST)
{
	for (const auto& e : TEST)
	{
		if (!isdigit(e))
			return false;
		if (e == '.')
			continue;
	}
	return true;
}


bool DataSet::isStr(const std::string _TEST_)
{
	for (const auto& e : _TEST_)
	{
		if (!(e >= 65 && e <= 90) || !(e >= 97 && e <= 122))
			return false;
	}
	return true;
}


DataSet::DataSet(std::vector<std::string>* cris)
{
	if (cris)
	{
		for (size_t i = 0; i < cris->size(); i++)
		{
			this->cris[(*cris)[i]] = (*cris)[i];
		}
	}
}



bool DataSet::loadCsvFile(const char filename[])
{

	FILE* file = fopen(filename, "r");
	if (!file)
		return false;
	char str[200];
	std::string row;
	int count = 0;
	while (fgets(str, sizeof(str), file))
	{
		rowNum++;
		char* temp = strtok(str, ",");
		PAIR* pair = new PAIR;
		int labelCount = static_cast<int>(labels.size() - 1);
		while (temp != NULL)
		{
			row = temp;
			if (count == 0)
			{
				labels.push_back(row);
			}
			else
			{
				if (labelCount > 0)
					pair->first.push_back(row);
				else
				{
					pair->second = row;
				}
				labelCount--;
			}
			temp = strtok(NULL, ",");
		}
		if (count != 0)
			this->set.push_back(*pair);
		count++;
	}
	for (int i = 0; i < labels.size(); i++)
	{
		if (i == labels.size() - 1)
			labels[i] = labels[i].substr(0, labels[i].size() - 1);
		this->cris[labels[i]] = labels[i];

	}
	return true;
}


void DataSet::saniTize()
{
	for (const auto& pair : this->set)
	{
		if (!isFLoat(pair.second) || !isInt(pair.second) || !isStr(pair.second))
		{
			this->erasePairAt(pair.second);
			continue;
		}
		for (const auto& colum : pair.first)
		{
			if (!isFLoat(colum) || !isInt(colum) || !isStr(colum))
			{
				this->erasePairAt(colum);
				continue;
			}
		}
	}
}


void DataSet::display(DataSet* dataset, unsigned limit)
{
	std::string le;
	for (const auto& e : this->labels)
	{
		le += "|\t" + e + '\t';

	}
	int len = strlen(le.c_str());
	auto draw = [len]()
		{
			std::cout << " ";
			for (int i = 0; i <= len * 2.25; i++)
				std::cout << "-";
			std::cout << "\n";
		};

	draw();
	std::cout << le << "\n";
	draw();
	for (auto it = this->set.begin(); it != this->set.end(); ++it)
	{
		auto row = *it;
		for (const auto& label : row.first)
		{
			std::cout << "|\t" << label << "\t|";


		}
		std::cout << row.second << "";
		std::cout.clear();
		if (!limit)
			break;
		limit--;
	}
}





void rx::DataSet::printPair(const PAIR& pair)
{
	std::string le;
	for (const auto& e : this->labels)
	{
		le += "|\t" + e + '\t';

	}
	int len = strlen(le.c_str());
	auto draw = [len]()
		{
			std::cout << " ";
			for (int i = 0; i <= len * 2.25; i++)
				std::cout << "-";
			std::cout << "\n";
		};

	draw();
	std::cout << le << "\n";
	draw();
	for (const auto& label : pair.first)
	{
		if (label.size() < 5)
			std::cout << "|\t";
		else
			std::cout << "|";
		std::cout << label << "\t|";

	}
	std::cout << pair.second << "";
	std::cout << "\n\n";
}

PAIR* DataSet::getPairAt(std::string key)
{
	for (auto& e : this->set)
	{
		for (const auto& label : e.first)
		{
			int count = 0;
			if (count == 0)
			{
				if (label == key)
					return &e;
			}
			count++;
		}
	}
	return nullptr;
}


 void DataSet::erasePairAt(std::string key)
{
	for (auto it = this->set.begin(); it != this->set.end(); ++it)
	{
		auto& e = *it;
		for (const auto& label : e.first)
		{
			int count = 0;
			if (count == 0)
			{
				if (label == key)
				{
					it = this->set.erase(it);
					--it;
					return;
				}
			}
		}
	}
}


 void DataSet::clear()
{
	this->set.clear();
	this->labels.clear();
	this->cris.clear();
}


SET* DataSet::getSet()
{
	return &this->set;
}


PAIR* rx::DataSet::getPairAtIndex(int index)
{
	return &this->set[index];
}

std::vector<std::string> rx::DataSet::valueAtIndex(int index)
{
	return this->set[index].first;
}

void rx::DataSet::setSet(rx::SET& newSet)
{
	this->set = newSet;
}

std::vector<std::string>& DataSet::getLabels()
{
	return this->labels;
}


 std::string DataSet::getOutputLabel()
{
	return this->labels[this->labels.size() - 1];
}


 const int DataSet::getRowNum() const
{
	return this->rowNum;
}


 void DataSet::initFreqTable(const std::string _ou1, const std::string _out2)
{
	int Ecount = 0;
	for (const auto& label : this->labels)
	{
		if (label == this->getOutputLabel())
			return;
		std::vector<std::pair<std::string, int>> temp;
		FreqTable freq;
		for (const auto& e : this->set)
		{
			for (int i = 0; i < e.first.size(); i++)
			{
				const auto& row = e.first[i];
				if (i == Ecount)
				{
					std::pair<std::string, int> p;
					p.first = row;
					if (e.second == "Yes\n" || e.second == "1\n" || e.second == "Yes" || e.second == "1")
					{
						p.second = 1;
						temp.push_back(p);
					}
					else if (e.second == "No\n" || e.second == "0\n" || e.second == "No" || e.second == "0")
					{
						p.second = 0;
						temp.push_back(p);
					}
				}
			}
		}
		for (const auto& e : temp)
		{
			if (e.second == 1)
			{
				freq[e.first].first++;
			}
			else if (e.second == 0)
			{
				freq[e.first].second++;
			}
		}
		this->freq_tables[label] = freq;
		Ecount++;
	}
}


FreqTable* DataSet::getFreqTableAt(std::string key)
{
	return &this->freq_tables[key];
}


 void DataSet::printFreqTableAt(std::string key)
{
	if (this->freq_tables.find(key) == this->freq_tables.end())
	{
		std::cout << "Element does not exist.\n";
		exit(1);
	}
	std::string le;

	le += "|\t" + key + '\t';
	le += "|\t" + this->getOutputLabel() + '\t';
	le += "|\t Not" + this->getOutputLabel() + '\t';

	int len = strlen(le.c_str());
	auto draw = [len]()
		{
			std::cout << "\n" << " ";
			for (int i = 0; i <= len * 2.25; i++)
				std::cout << "-";
			std::cout << "\n";
		};

	draw();
	std::cout << le << "\n";
	draw();
	for (const auto& e : this->freq_tables[key])
	{
		std::cout << "|\t" << e.first;
		if (e.first.size() < 5)
			std::cout << "\t\t|";
		else
			std::cout << "\t|";
		std::cout << e.second.first << "\t\t|\t" << e.second.second << "\t\t\n";
	}
	std::cout << "\n";
}




 std::map<std::string, int> DataSet::getOutputAnalisis()
{


	std::map<std::string, int> temp;
	for (const auto& table : this->set)
	{
		const std::string str = cleanStr(table.second);
		temp[str]++;

	}
	return temp;
}

 void DataSet::describe(short TYPE, std::string _LABEL_)
{

	switch (TYPE)
	{
	case AVG:
		if (_LABEL_ == "")
		{
			this->describeAvg();
			break;
		}
		this->describeLabel(_LABEL_);
		break;
	case LABEL:
	default:
		this->describeAvg();
		break;
	};

}


 std::string DataSet::getAvgAtColumn(std::string _COLUMN_)
{
	PAIR temp;
	auto findMostFreq = [](const std::map<std::string, float>& test) {
		int max = 0;
		for (const auto& e : test)
		{
			if (e.second > max)
				max = e.second;
		}
		for (const auto& e : test)
		{
			if (e.second == max)
				return e.first;
		}
		return std::string("Nan");
		};
	float totalF = 0; int totalInt = 0;
	std::map<std::string, std::map<std::string, float>> strs;
	int count = 0;
	while (count < this->labels.size() - 1)
	{

		for (const auto& Table : this->set)
		{
			int localCount = 0;
			for (const auto& row : Table.first)
			{
				if (count == localCount)
				{
					if (isFLoat(row))
						strs[this->labels[count]][row] += atof(row.c_str());
					else if (isInt(row))
					{
						strs[this->labels[count]][row] += atoi(row.c_str());
					}
					else
					{

						strs[this->labels[count]][row]++;
					}
				}
				localCount++;
			}

		}
		count++;
	}

	return findMostFreq(strs[_COLUMN_]);

}


 void DataSet::info()
{
	std::cout << "(" << this->getRowNum() << ", " << this->getLabels().size() << ")\n";
}


 void DataSet::describeAvg()
{
	PAIR temp;
	auto isFLoat = [](const std::string& test)
		{
			for (const auto& e : test)
			{
				if (e == '.')
					return true;
			}
			return false;
		};
	auto isInt = [](const std::string& test)
		{
			for (const auto& e : test)
				if (e <= '0' || e >= '9')
					return false;
			return true;
		};
	auto findMostFreq = [](const std::map<std::string, float>& test) {
		int max = 0;
		for (const auto& e : test)
		{
			if (e.second > max)
				max = e.second;
		}
		for (const auto& e : test)
		{
			if (e.second == max)
				return e.first;
		}
		return std::string("Nan");
		};
	float totalF = 0; int totalInt = 0;
	std::map<std::string, std::map<std::string, float>> strs;
	int count = 0;
	while (count < this->labels.size() - 1)
	{

		for (const auto& Table : this->set)
		{
			int localCount = 0;
			for (const auto& row : Table.first)
			{
				if (count == localCount)
				{
					if (isFLoat(row))
						strs[this->labels[count]][row] += atof(row.c_str());
					else if (isInt(row))
					{
						strs[this->labels[count]][row] += atoi(row.c_str());
					}
					else
					{

						strs[this->labels[count]][row]++;
					}
				}
				localCount++;
			}

		}
		count++;
	}

	std::string le;
	le += "\t";
	int unitlen = 0;


	for (const auto& e : this->labels)
	{
		if (e == this->getOutputLabel())
			break;
		std::string temp("|\t" + e + '\t');
		le += temp;
		if (unitlen < temp.size())
			unitlen = temp.size();
	}


	int len = strlen(le.c_str());
	auto draw = [len]()
		{
			std::cout << " ";
			for (int i = 0; i <= len * 2.5; i++)
				std::cout << "-";
			std::cout << "\n";
		};
	auto cleanStr = [](std::string str)
		{
			if (str[str.size() - 1] == '\n')
				str = str.substr(0, str.size() - 1);
			return str;
		};

	draw();
	std::cout << le << "\n";
	draw();

	std::cout << "|Avg  |";
	for (int i = 0; i < this->labels.size() - 1; i++)
	{
		std::string temp = cleanStr(findMostFreq(strs[this->labels[i]]));
		int j;
		std::cout << "|\t";

		for (j = 0; j < unitlen; j++)
		{
			if (j == tolower((unitlen / 2.f) - (temp.size() / 2.f)))
			{
				std::cout << temp;
				break;
			}
			std::cout << " ";
		}


		for (int k = 0; k < unitlen - (j + temp.size()) + 3; k++)
		{
			std::cout << " ";
		}
		std::cout << "|";
	}
}


 void DataSet::describeLabel(std::string _LABEL_)
{
	if (!this->findInVec(this->getLabels(), _LABEL_))
	{
		std::cout << "Label is invalid.\n";
		exit(EXIT_FAILURE);
	}


	int len = (int)strlen("||") + (int)strlen(_LABEL_.c_str()) + 16;
	auto draw = [](int len) {for (int i = 0; i < len - 1; i++) printf("="); printf("\n"); };
	draw(len);
	printf("|\t%s\t|\n", _LABEL_);
	draw(len);
	printf("|\t%*s\t|", strlen(_LABEL_.c_str()), this->getAvgAtColumn(_LABEL_));
	printf("\n");

}