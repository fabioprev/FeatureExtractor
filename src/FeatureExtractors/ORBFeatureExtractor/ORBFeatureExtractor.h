#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <mutex>

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
			 * @brief mutex to synchronise gif extraction and descriptor generation.
			 */
			std::mutex mutex;
			
			/**
			 * @brief semaphore used to safely run multiple instances of 'convert'.
			 */
			long semaphoreId;
			
			/**
			 * @brief \b true means that the gif extraction is running, \b false otherwise.
			 */
			bool isExtracting;
			
			/**
			 * @brief Function that invokes a thread-function that checks the number of running instances of 'convert'.
			 * 
			 * @param orbFeatureDetector pointer to the invocation object.
			 * 
			 * @return 0 if succeeded, -1 otherwise.
			 */
			static void* checkerThread(ORBFeatureExtractor* orbFeatureExtractor) { orbFeatureExtractor->checker(); return 0; }
			
			/**
			 * @brief Function that checks the number of running instances of 'convert'.
			 */
			void checker();
			
			/**
			 * @brief Function that creates a semaphore.
			 */
			void createSemaphore();
			
			/**
			 * @brief Function that starts the extraction of the keypoints and of the descriptors of the image given as input.
			 * 
			 * @param image reference to the image.
			 * @param outputDirectory name of the directory where generated images will be saved.
			 * @param section number of the section of the patient.
			 */
			void exec(const cv::Mat& image, const std::string& outputDirectory, const std::string& section);
			
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
			 * 
			 * @return a vector containing all gif images in the directory given as input.
			 */
			std::vector<std::string> extractFramesFromGif(const std::string& directory);
			
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
