#include "caffe/util/random_helper.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <time.h>
#include <math.h>

namespace caffe {
	random_helper::random_init_helper random_helper::m_randInit;

#if _HAS_CPP0X
	std::mt19937_64 random_helper::m_generator;
	std::uniform_int_distribution<unsigned int> random_helper::m_uniformInt;
	std::uniform_real_distribution<double> random_helper::m_uniformReal;
	std::normal_distribution<double> random_helper::m_normalReal;
#endif

	random_helper::random_init_helper::random_init_helper() {
		srand(time(NULL));
#if _HAS_CPP0X
		m_generator = std::mt19937_64(std::random_device()());
		m_uniformInt = std::uniform_int_distribution<unsigned int>(0, UINT_MAX);
		m_uniformReal = std::uniform_real_distribution<double>(0.0, 1.0);
		m_normalReal = std::normal_distribution<double>(0.0, 1.0);
#endif
	}

	random_helper::random_helper() {
	}


	random_helper::~random_helper() {
	}


	unsigned int random_helper::uniform_int(unsigned int min/* = 0*/, unsigned int max/* = UINT_MAX*/) {
		if (min >= max) {
			std::cout << "random_helper::uniform_int must have min >= max" << std::endl;
			throw std::runtime_error("random_helper::uniform_int must have min >= max");
		}

		unsigned int range = max - min + 1;

#ifndef RAND_DEVICE
#if _HAS_CPP0X
		if (m_uniformInt.min() != min || m_uniformInt.max() != max)
			m_uniformInt = std::uniform_int_distribution<unsigned int>(min, max);
		return m_uniformInt(m_generator);
#else
		if (max > RAND_MAX)
			std::cout << "Warning : random_helper::uniform_int with rand() only with range [0, " << RAND_MAX << "]" << std::endl;
		return rand() % range + min;
#endif
#else
		if (max > std::random_device().max())
			std::cout << "Warning : random_helper::uniform_int with random_device() only with range [0, " << std::random_device().max() << "]" << std::endl;
		return std::random_device()() % range + min;
#endif
	}

	double random_helper::uniform_real(double min/*= 0.0*/, double max/* = 1.0*/) {
		if (min >= max) {
			std::cout << "random_helper::uniform_real must have min >= max" << std::endl;
			throw std::runtime_error("random_helper::uniform_real must have min >= max");
		}

#ifndef RAND_DEVICE
#if _HAS_CPP0X
		if (m_uniformReal.min() != min || m_uniformReal.max() != max)
			m_uniformReal = std::uniform_real_distribution<double>(min, max);
		return m_uniformReal(m_generator);
#else
		return (double)rand() / (double)RAND_MAX * (max - min) + min;
#endif
#else
		return (double)std::random_device()() / (double)std::random_device().max() * (max - min) + min;
#endif
	}


	double random_helper::normal_real()
	{
#ifndef RAND_DEVICE
#if _HAS_CPP0X
		return m_normalReal(m_generator);
#else
		const double PI = 3.1415926535897932384626433832795;
		double a = ((double)rand() + 1) / ((double)RAND_MAX + 2);
		double b = (double)rand() / ((double)RAND_MAX + 1);
		return sqrt(-2 * log(a))*cos(2 * PI*b);

#endif
#else
		const double PI = 3.1415926535897932384626433832795;
		double a = ((double)std::random_device()() + 1) / ((double)std::random_device().max() + 2);
		double b = (double)std::random_device()() / ((double)std::random_device().max() + 1);
		return sqrt(-2 * log(a))*cos(2 * PI*b);
#endif
	}
}
