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
			ImageDescriptor = 0,
			Histograms,
			ImageDescriptorAndHistograms,
			HashCantor,
			HistogramsAndHashCantor,
			Unknown
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
		 * @brief Function that returns the strategy chosen.
		 * 
		 * @param strategy strategy chosen in a string format.
		 * 
		 * @return the strategy chosen.
		 */
		inline static CSVGenerationStrategy getStrategy(const std::string& strategy)
		{
			if (strcasecmp(strategy.c_str(),"ImageDescriptor") == 0) return Utils::ImageDescriptor;
			else if (strcasecmp(strategy.c_str(),"Histograms") == 0) return Utils::Histograms;
			else if (strcasecmp(strategy.c_str(),"ImageDescriptorAndHistograms") == 0) return Utils::ImageDescriptorAndHistograms;
			else if (strcasecmp(strategy.c_str(),"HashCantor") == 0) return Utils::HashCantor;
			else if (strcasecmp(strategy.c_str(),"HistogramsAndHashCantor") == 0) return Utils::HistogramsAndHashCantor;
			else return Utils::Unknown;
		}
		
		/**
		 * @brief Function that returns the strategy chosen in a string format.
		 * 
		 * @param strategy strategy chosen.
		 * 
		 * @return the strategy chosen in a string format.
		 */
		inline static std::string getStrategyString(const CSVGenerationStrategy& strategy)
		{
			if (strategy == Utils::ImageDescriptor) return "ImageDescriptor";
			else if (strategy == Utils::Histograms) return "Histograms";
			else if (strategy == Utils::ImageDescriptorAndHistograms) return "ImageDescriptorAndHistograms";
			else if (strategy == Utils::HashCantor) return "HashCantor";
			else if (strategy == Utils::HistogramsAndHashCantor) return "HistogramsAndHashCantor";
			else return "Unknown";
		}
		
		/**
		 * @brief Function that checks whether the number given as input is actually a number.
		 * 
		 * @param number the number to be checked.
		 * 
		 * @return \b true if the number given as input is actually a number, \b false otherwise.
		 */
		inline static bool isNotNumber(const std::string& number)
		{
			for (unsigned int i = 0; i < number.size(); ++i)
			{
				if (!isdigit(number[i])) return true;
			}
			
			return false;
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
