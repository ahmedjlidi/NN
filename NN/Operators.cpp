#include "stdafx.h"
#include "Operators.h"

// Definitions of the operator<< functions
std::ostream& operator<<(std::ostream& os, const std::vector<float>& vec) {
    os << "[ ";
    for (size_t j = 0; j < vec.size(); ++j) {
        os << vec[j];
        if (j + 1 < vec.size()) {
            os << ", ";
        }
    }
    os << " ]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<float>>& v) {
    for (const auto& row : v) {
        os << "[ ";
        for (size_t j = 0; j < row.size(); j++) {
            os << row[j];
            if (j + 1 < row.size()) {
                os << ", ";
            }
        }
        os << "]\n";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::pair<int, int> v) {
    os << "(" << v.first << ", " << v.second << ")\n";
    return os;
}
