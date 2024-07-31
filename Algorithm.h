#pragma once
#include "DataSet.h"
#include "Utility.h"


class Algorithm
{
	friend class Utility;
protected:

	
	rx::DataSet* dataset;
	float percentage;
	rx::SET* set;


public:
	Algorithm(){}
	Algorithm(rx::SET* set);
	Algorithm(rx::DataSet* dataset, float percentage = 100);
	virtual void buildModel() {}
	virtual std::string predict(std::vector<std::string>) { return "null"; }
	virtual std::string predict(std::vector <std::pair<std::string, std::string>>& v) { return "null"; }
	virtual const float getPercentage() const;
	rx::DataSet& getSet();
};

