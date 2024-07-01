#ifndef OPERATORS_H
#define OPERATORS_H

#include <iostream>
#include <vector>
#include <utility>

// Declarations of the operator<< functions
std::ostream& operator<<(std::ostream& os, const std::vector<float>& vec);
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<float>>& v);
std::ostream& operator<<(std::ostream& os, const std::pair<int, int> v);
#endif
