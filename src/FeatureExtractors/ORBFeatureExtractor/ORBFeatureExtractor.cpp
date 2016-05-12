#include "ORBFeatureExtractor.h"

using namespace std;
using namespace cv;

namespace FeatureExtractors
{
	ORBFeatureExtractor::ORBFeatureExtractor() {;}
	
	ORBFeatureExtractor::~ORBFeatureExtractor() {;}
	
	void ORBFeatureExtractor::exec()
	{
		exec("/home/fabio/TEST2/section_052.png");
	}
	
	void ORBFeatureExtractor::exec(const string& filename)
	{
		Mat image = imread(filename);
		
		const Mat& keypointsImage = extractKeyPoints(image);
		const Mat& descriptorsImage = extractDetectors(image);
		
		imshow("Key points",keypointsImage);
		imshow("Descriptors",descriptorsImage);
		
		waitKey(0);
	}
	
	Mat ORBFeatureExtractor::extractDetectors(const Mat& image)
	{
		Mat descriptorsImage;
		
		descriptorExtractor.compute(image,keypoints,descriptorsImage);
		
		return descriptorsImage;
	}
	
	Mat ORBFeatureExtractor::extractKeyPoints(const Mat& image)
	{
		featureDetector.detect(image,keypoints);
		
		Mat keypointsImage;
		
		drawKeypoints(image,keypoints,keypointsImage,Scalar::all(255));
		
		return keypointsImage;
	}
}
