#include "ORBFeatureExtractor.h"
#include <Utils/DebugUtils.h>
#include <sys/stat.h>
#include <fstream>

using namespace std;
using namespace cv;

namespace FeatureExtractors
{
	ORBFeatureExtractor::ORBFeatureExtractor() {;}
	
	ORBFeatureExtractor::~ORBFeatureExtractor() {;}
	
	void ORBFeatureExtractor::exec(const string& directory)
	{
		extractFramesFromGif(directory);
	}
	
	void ORBFeatureExtractor::exec(Mat& image)
	{
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
	
	void ORBFeatureExtractor::extractFramesFromGif(const string& directory)
	{
		vector<string> images;
		ifstream imageFiles;
		char buffer[4096];
		
		if (system((string("find ") + directory + string(" -name *.gif > .temp")).c_str()));
		
		imageFiles.open(".temp");
		
		while (imageFiles.good())
		{
			if (imageFiles.eof()) break;
			
			imageFiles.getline(buffer,4096);
			
			if (strlen(buffer) > 0) images.push_back(buffer);
		}
		
		imageFiles.close();
		
		if (system("rm -rf .temp"));
		
		for (vector<string>::const_iterator it = images.begin(); it != images.end(); ++it)
		{
			struct stat status;
			bool exists;
			
			exists = false;
			
			if ((stat((it->substr(0,it->rfind("/")) + string("/sections")).c_str(),&status) == 0) && S_ISDIR(status.st_mode)) exists = true;
			
			INFO("Extracting frames from '");
			WARN(*it);
			INFO("'...");
			
			if (system((string("cd ") + it->substr(0,it->rfind("/")) + string(" && ") + (!exists ? string("mkdir sections && ") : string("")) + string("cd sections && convert ") + *it + string(" -coalesce section_%03d.png")).c_str()));
			
			INFO("done!" << endl);
		}
	}
}
