#include "stdafx.h"
#include "Algorithm.h"

Algorithm::Algorithm(rx::DataSet* dataset, float percentage) : dataset(dataset), percentage(percentage)
{
	const float lim = (this->dataset->getRowNum() * this->percentage) / static_cast<float>(100.f);
	rx::SET* set = new rx::SET;
	rx::SET* s = dataset->getSet();
	for (int i = 0; i < lim + 1 && i <s->size(); i++)
	{
		set->push_back((*s)[i]);
	}
	dataset->setSet(*set);
}

Algorithm::Algorithm(rx::SET* set): set(set)
{
}


const float Algorithm::getPercentage() const
{
	return this->percentage;
}

rx::DataSet& Algorithm::getSet()
{
	return *this->dataset;
}
