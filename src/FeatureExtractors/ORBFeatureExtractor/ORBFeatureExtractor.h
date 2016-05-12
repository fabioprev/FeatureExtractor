#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace FeatureExtractors
{
	/**
	 * @class ORBFeatureExtractor
	 * 
	 * @brief Class that implements a ORB features extractor.
	 */
	class ORBFeatureExtractor
	{
		private:
			/**
			 * @brief vector of keypoints found in the image.
			 */
			std::vector<cv::KeyPoint> keypoints;
			
			/**
			 * @brief descriptor extractor based on ORB.
			 */
			cv::OrbDescriptorExtractor descriptorExtractor;
			
			/**
			 * @brief feature extractor based on ORB.
			 */
			cv::OrbFeatureDetector featureDetector;
			
			/**
			 * @brief Function that starts the extraction of the keypoints and of the descriptors of the image given as input.
			 * 
			 * @param image reference to the image.
			 */
			void exec(cv::Mat& image);
			
			/**
			 * @brief Function that returns the descriptors of the image given the keypoints.
			 * 
			 * @param image reference to the image to be analysed.
			 * 
			 * @return an image representing the descriptors of the image given the keypoints.
			 */
			cv::Mat extractDetectors(const cv::Mat& image);
			
			/**
			 * @brief Function that returns the keypoints of the image.
			 * 
			 * @param image reference to the image to be analysed.
			 * 
			 * @return an image representing the keypoints of the image.
			 */
			cv::Mat extractKeyPoints(const cv::Mat& image);
			
			/**
			 * @brief Function that extracts frames of all gif images in the directory given as input.
			 * 
			 * @param directory name of the directory which contains the images to be analysed.
			 */
			void extractFramesFromGif(const std::string& directory);
			
		public:
			/**
			 * @brief Empty constructor.
			 */
			ORBFeatureExtractor();
			
			/**
			 * @brief Destructor.
			 */
			~ORBFeatureExtractor();
			
			/**
			 * @brief Function that starts the extraction of the keypoints and of the descriptors of the image contained (recursively) in the directory given as input.
			 * 
			 * @param directory name of the directory which contains the images to be analysed.
			 */
			void exec(const std::string& directory);
	};
}
