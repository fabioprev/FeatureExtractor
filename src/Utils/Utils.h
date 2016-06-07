#pragma once

#include <opencv2/features2d/features2d.hpp>
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
		 * @brief Enumerator representing all the possible strategy to generate the CSV file.
		 */
		enum CSVGenerationStrategy
		{
			ImageDescriptors = 0,
			Histograms,
			ImageDescriptorsAndHistograms
		};
		
		/**
		 * @brief Function that compares two keypoints.
		 * 
		 * @param i reference to the first keypoint to be compared.
		 * @param j reference to the second keypoint to be compared.
		 * 
		 * @return \b true if the first keypoint is greater than the second one, \b false otherwise.
		 */
		inline static bool compareKeyPoint(const cv::KeyPoint& i, const cv::KeyPoint& j)
		{
			return (i.response > j.response);
		}
		
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
