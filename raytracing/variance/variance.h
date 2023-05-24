#pragma once

// incremental variance

#include <cmath>

template<typename _val>
class VarianceObject {
	int n;
	_val sx, sx2;
public:
	VarianceObject() {
		n = 0;
		sx = sx2 = _val(0.0);
	}
	void addElement(_val x) {
		n += 1;
		sx += x;
		sx2 += x * x;
	}
	int getN() const {
		return n;
	}
	_val getMean() const {
		return sx / _val(n);
	}
	_val getVariance() const {
		return (sx2 - sx * sx / _val(n)) / _val(n - 1);
	}
	_val getStandardDeviation() const {
		return sqrt(this->getVariance());
	}
	_val getRandomError() const {
		return sqrt(this->getVariance() / _val(n));
	}
};
