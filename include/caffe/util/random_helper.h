
#pragma once

#if _HAS_CPP0X
#include <random>
#endif
#include <climits>

namespace caffe {
	class random_helper {
	private:
		class random_init_helper {
		public:
			random_init_helper();
		};

	private:
		random_helper();
		~random_helper();

	public:
		static unsigned int uniform_int(unsigned int min = 0, unsigned int max = UINT_MAX);
		static double uniform_real(double min = 0.0, double max = 1.0);
		static double normal_real();

	private:
		static random_init_helper m_randInit;

#if _HAS_CPP0X
		static std::mt19937_64 m_generator;
		static std::uniform_int_distribution<unsigned int> m_uniformInt;
		static std::uniform_real_distribution<double> m_uniformReal;
		static std::normal_distribution<double> m_normalReal;
#endif
	};
}
