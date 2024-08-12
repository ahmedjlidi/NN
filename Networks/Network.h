#ifndef NETWORK_H
#define NETWORK_H

#include "Resources/Net.h"
#include "BNet.h"
#include "MNet.h"
#include "CNet.h"
#include "../stdafx.h"


namespace rx
{
	enum TYPE : short
	{
		REGRESSION,
		BINARY_CLASSIFICATION,
		CROSS_CLASSIFICATION,
	};
	std::unique_ptr<Net> initNet(short type)
	{
		switch (type)
		{
		case BINARY_CLASSIFICATION:
			return std::make_unique<BNet>();
			break;
		case REGRESSION:
			return std::make_unique<MNet>();
			break;
		case CROSS_CLASSIFICATION:
			return std::make_unique<CNet>();
			break;
		default:
			std::cerr << "Type is not defined.\n";
			exit(1);
		}
	}
}

#endif