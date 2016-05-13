#pragma once

#include <math.h>

/**
 * @class Utils
 * 
 * @brief Class that defines several useful functions.
 */
class Utils
{
	public:
		/**
		 * @brief Function that truncates a floating number to a specified decimal position.
		 * 
		 * @param d floating number to be truncated.
		 * @param n desired decimal position.
		 * 
		 * @return the truncated floating number.
		 */
		inline static float roundN(float d, int n)
		{
			if (n == 0) return round(d);
			else if (n > 0)
			{
				float p;
				int temp;
				
				p = std::pow(10.0,n);
				temp = (int) (d * p);
				
				return (((float) temp) / p);
			}
			else return d;
		}
};
