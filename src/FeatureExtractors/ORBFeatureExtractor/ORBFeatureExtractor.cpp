#include "ORBFeatureExtractor.h"
#include <Utils/ConfigFile.h>
#include <Utils/DebugUtils.h>
#include <Utils/Utils.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace cv;

namespace FeatureExtractors
{
	ORBFeatureExtractor::ORBFeatureExtractor() : isExtracting(false)
	{
		ConfigFile fCfg;
		stringstream s;
		string key, section, temp;
		
		if (!fCfg.read("../config/parameters.cfg"))
		{
			ERR("Error reading file '" << "../config/parameters.cfg" << "' for ORBFeatureExtractor configuration. Exiting..." << endl);
			
			exit(-1);
		}
		
		try
		{
			section = "Dataset";
			
			key = "name";
			dataset = string(fCfg.value(section,key));
			
			section = "FeatureExtractor";
			
			key = "histogramHorizontalBins";
			histogramHorizontalBins = fCfg.value(section,key);
			
			key = "histogramVerticalBins";
			histogramVerticalBins = fCfg.value(section,key);
			
			key = "maxFeatureNumber";
			maxFeatureNumber = fCfg.value(section,key);
			
			key = "strategy";
			strategy = Utils::getStrategy(string(fCfg.value(section,key)));
			
			section = "Brain";
			
			key = "sections";
			temp = string(fCfg.value(section,key));
			
			s << temp;
			
			while (s.good())
			{
				if (s.eof()) break;
				
				getline(s,temp,',');
				
				sections.push_back(temp.c_str());
			}
			
			ERR("******************************************************" << endl);
			DEBUG("Feature extractor parameters:" << endl);
			
			INFO("\tData set: ");
			WARN(dataset << endl);
			
			INFO("\tHorizontal histogram bins: ");
			WARN(histogramHorizontalBins << endl);
			
			INFO("\tVertical histogram bins: ");
			WARN(histogramVerticalBins << endl);
			
			INFO("\tMaximum number of features: ");
			WARN(maxFeatureNumber << endl);
			
			INFO("\tFeature selection strategy: ");
			WARN(Utils::getStrategyString(strategy) << endl);
			
			INFO("\tPatient brain sections: ");
			
			if (strcasecmp(dataset.c_str(),"OASIS") == 0) WARN("000" << endl)
			else
			{
				for (vector<string>::const_iterator it = sections.begin(); it != sections.end(); ++it)
				{
					WARN(*it);
					
					if ((it + 1) != sections.end()) WARN(",");
				}
				
				WARN(endl);
			}
			
			ERR("******************************************************" << endl << endl);
		}
		catch (...)
		{
			ERR("Not existing value '" << section << "/" << key << "'. Exiting..." << endl);
			
			exit(-1);
		}
	}
	
	ORBFeatureExtractor::~ORBFeatureExtractor() {;}
	
	void ORBFeatureExtractor::checker()
	{
		ifstream runningInstancesFile;
		int numberOfCores = 1, ret, runningInstances;
		char buffer[128];
		
		/// Waiting 'convert' instances start running.
		usleep(2e6);
		
		if (system("nproc > .temp"));
		
		runningInstancesFile.open(".temp");
		
		while (runningInstancesFile.good())
		{
			if (runningInstancesFile.eof()) break;
			
			runningInstancesFile.getline(buffer,128);
			
			numberOfCores = atoi(buffer);
			
			break;
		}
		
		mutex.lock();
		
		runningInstances = 0;
		
		while (isExtracting)
		{
			if (system("ps -A | grep convert | wc -l > .temp"));
			
			runningInstancesFile.open(".temp");
			
			while (runningInstancesFile.good())
			{
				if (runningInstancesFile.eof()) break;
				
				runningInstancesFile.getline(buffer,128);
				
				runningInstances = atoi(buffer);
				
				if (runningInstances < numberOfCores)
				{
					struct sembuf oper;
					
					oper.sem_num = 0;
					oper.sem_op = numberOfCores - runningInstances;
					oper.sem_flg = 0;
					
					ret = semop(semaphoreId,&oper,1);
					
					if (ret == -1)
					{
						WARN("An error occured when calling semop. I am not exiting though..." << endl);
					}
				}
				
				break;
			}
			
			runningInstancesFile.close();
			
			if (system("rm -rf .temp"));
			
			usleep(500e3);
		}
		
		while (runningInstances > 0)
		{
			if (system("ps -A | grep convert | wc -l > .temp"));
			
			runningInstancesFile.open(".temp");
			
			while (runningInstancesFile.good())
			{
				if (runningInstancesFile.eof()) break;
				
				runningInstancesFile.getline(buffer,128);
				
				runningInstances = atoi(buffer);
				
				break;
			}
			
			runningInstancesFile.close();
			
			if (system("rm -rf .temp"));
			
			usleep(100e3);
		}
		
		mutex.unlock();
	}
	
	void ORBFeatureExtractor::createSemaphore()
	{
		long semaphoreHandle = 31;
		int ret;
		
		semaphoreId = semget(semaphoreHandle,1,IPC_CREAT | 0666);
		
		if (semaphoreId == -1)
		{
			ERR("Impossible to create the semaphore. Exiting..." << endl);
			
			exit(-1);
		}
		
		ret = semctl(semaphoreId,0,SETVAL,8);
		
		if (ret == -1)
		{
			ERR("An error occured when calling semctl. Exiting ..." << endl);
			
			exit(-1);
		}
	}
	
	void ORBFeatureExtractor::exec(const string& directory)
	{
		int counter;
		
		const vector<string>& images = extractFramesFromGif(directory);
		
		/// Synchronising with gif thread checker. This code must be executed only when the thread has finished is execution.
		mutex.lock();
		mutex.unlock();
		
		counter = 0;
		
		for (vector<string>::const_iterator it = images.begin(); it != images.end(); ++it)
		{
			stringstream s;
			string patientPath;
			
			patientPath = it->substr(0,it->rfind("/")) + string("/sections/");
			
			if (system((string("rm -rf ") + patientPath + string("../features/keypoints")).c_str()));
			if (system((string("rm -rf ") + patientPath + string("../features/descriptors")).c_str()));
			
			if (system((string("mkdir -p ") + patientPath + string("../features/keypoints")).c_str()));
			if (system((string("mkdir -p ") + patientPath + string("../features/descriptors")).c_str()));
			
			s << setw(3) << setfill(' ') << Utils::roundN(counter++ / (float) images.size() * 100,0);
			
			ERR("[" << s.str() << "%] ");
			INFO("Generating features of '");
			WARN(it->substr(0,it->rfind("/")));
			INFO("'...");
			
			for (vector<string>::const_iterator it2 = sections.begin(); it2 != sections.end(); ++it2)
			{
				const Mat& image = imread(patientPath + string("section_") + *it2 + string(".png"));
				
				if (!image.empty()) exec(image,patientPath,string("section_") + *it2 + string(".png"));
			}
			
			INFO("done!" << endl);
		}
		
		writeJointCSV(directory,sections);
	}
	
	void ORBFeatureExtractor::exec(const Mat& image, const string& outputDirectory, const string& section)
	{
		struct stat status;
		
		Mat descriptorsImage, keypointsImage;
		bool isFeaturesGenerationNeeded;
		
		const Mat& descriptor = imread(outputDirectory + string("../features/descriptors/descriptor_") + section);
		
		/// Checking whether the current maximum number of features is different with respect to the one used to generate the feature's images, if any.
		if (!descriptor.empty() && descriptor.rows != maxFeatureNumber) isFeaturesGenerationNeeded = true;
		else isFeaturesGenerationNeeded = false;
		
		if (isFeaturesGenerationNeeded || (stat((outputDirectory + string("../features/keypoints/keypoint_") + section).c_str(),&status) != 0) || !S_ISREG(status.st_mode))
		{
			keypointsImage = extractKeyPoints(image);
		}
		
		if (isFeaturesGenerationNeeded || (stat((outputDirectory + string("../features/descriptors/descriptor_") + section).c_str(),&status) != 0) || !S_ISREG(status.st_mode))
		{
			descriptorsImage = extractDetectors(image);
		}
		
		if (!keypointsImage.empty() && !descriptorsImage.empty())
		{
			imwrite(outputDirectory + string("../features/keypoints/keypoint_") + section,keypointsImage);
			imwrite(outputDirectory + string("../features/descriptors/descriptor_") + section,descriptorsImage);
			
			writeCSV(image,descriptorsImage,outputDirectory + string("../features/descriptors/descriptor_") + section.substr(0,section.find(".png")));
		}
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
		
		sort(keypoints.begin(),keypoints.end(),Utils::compareKeyPoint);
		
		if ((int) keypoints.size() > maxFeatureNumber) keypoints.erase(keypoints.begin() + maxFeatureNumber,keypoints.end());
		
		Mat keypointsImage;
		
		drawKeypoints(image,keypoints,keypointsImage,Scalar::all(255),DrawMatchesFlags::DEFAULT);
		
		return keypointsImage;
	}
	
	vector<string> ORBFeatureExtractor::extractFramesFromGif(const string& directory)
	{
		vector<string> images;
		pthread_t threadId;
		ifstream imageFiles;
		int counter, ret;
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
		
		counter = 0;
		
		for (vector<string>::const_iterator it = images.begin(); it != images.end(); ++it)
		{
			struct stat status;
			
			if ((stat((it->substr(0,it->rfind("/")) + string("/sections")).c_str(),&status) == 0) && S_ISDIR(status.st_mode)) ++counter;
		}
		
		createSemaphore();
		
		isExtracting = true;
		
		pthread_create(&threadId,0,(void*(*)(void*)) checkerThread,this);
		
		for (vector<string>::const_iterator it = images.begin(); it != images.end(); ++it)
		{
			stringstream s;
			struct stat status;
			
			/// Sections have been already generated.
			if ((stat((it->substr(0,it->rfind("/")) + string("/sections")).c_str(),&status) == 0) && S_ISDIR(status.st_mode)) continue;
			
			struct sembuf oper;
			
			oper.sem_num = 0;
			oper.sem_op = -1;
			oper.sem_flg = 0;
			
			ret = semop(semaphoreId,&oper,1);
			
			if (ret == -1)
			{
				WARN("An error occured when calling semop. I am not exiting though..." << endl);
			}
			
			s << setw(3) << setfill(' ') << Utils::roundN(counter++ / (float) images.size() * 100,0);
			
			ERR("[" << s.str() << "%] ");
			INFO("Extracting frames from '");
			WARN(*it);
			INFO("'...");
			
			if (ret = system((string("cd ") + it->substr(0,it->rfind("/")) + string(" && mkdir -p sections && cd sections && convert ") + *it + string(" -coalesce section_%03d.png &")).c_str()));
			
			if (ret != 0) break;
			
			INFO("done!" << endl);
		}
		
		isExtracting = false;
		
		return images;
	}
	
	void ORBFeatureExtractor::writeCSV(const Mat& image, const Mat& descriptorsImage, const string& filename)
	{
		ofstream file;
		
		file.open((filename + string(".csv")).c_str());
		
		if ((strategy == Utils::ImageDescriptor) || (strategy == Utils::ImageDescriptorAndHistograms))
		{
			for (int i = 0; i < descriptorsImage.rows; ++i)
			{
				for (int j = 0; j < descriptorsImage.cols; ++j)
				{
					file << (int) descriptorsImage.at<uchar>(i,j);
					
					if ((strategy == Utils::ImageDescriptor) && (i == (descriptorsImage.rows - 1)) && (j == (descriptorsImage.cols - 1))) continue;
					
					file << ",";
				}
			}
		}
		
		if ((strategy == Utils::Histograms) || (strategy == Utils::ImageDescriptorAndHistograms) || (strategy == Utils::HistogramsAndHashCantor))
		{
			int featureHistogram[histogramHorizontalBins][histogramVerticalBins];
			
			/// Initializing the histogram's bins.
			for (int i = 0; i < histogramHorizontalBins; ++i)
			{
				for (int j = 0; j < histogramVerticalBins; ++j)
				{
					featureHistogram[i][j] = 0;
				}
			}
			
			for (vector<KeyPoint>::const_iterator it = keypoints.begin(); it != keypoints.end(); ++it)
			{
				++featureHistogram[(int) (it->pt.y / (image.rows / histogramHorizontalBins))][(int) (it->pt.x / (image.cols / histogramVerticalBins))];
			}
			
			for (int i = 0; i < histogramHorizontalBins; ++i)
			{
				for (int j = 0; j < histogramVerticalBins; ++j)
				{
					file << featureHistogram[i][j];
					
					if (((strategy == Utils::Histograms) || (strategy == Utils::ImageDescriptorAndHistograms)) && (i == (histogramHorizontalBins - 1)) && (j == (histogramVerticalBins - 1))) continue;
					
					file << ",";
				}
			}
		}
		
		if ((strategy == Utils::HashCantor) || (strategy == Utils::HistogramsAndHashCantor))
		{
			for (vector<KeyPoint>::const_iterator it = keypoints.begin(); it != keypoints.end(); ++it)
			{
				file << ((0.5 * (it->pt.x + it->pt.y) * (it->pt.x + it->pt.y + 1)) + it->pt.y);
				
				if ((it + 1) == keypoints.end()) continue;
				
				file << ",";
			}
		}
		
		file << endl;
		
		file.close();
	}
	
	void ORBFeatureExtractor::writeJointCSV(const string& directory, const vector<string>& sections)
	{
		vector<string> classes, csvFiles, xmlFiles;
		ofstream patientsFile;
		ifstream file;
		string date, oldPatient, patient, period, temp, temp2;
		int counter;
		char buffer[65536];
		
		classes.push_back("AD");
		classes.push_back("LMCI");
		classes.push_back("MCI");
		classes.push_back("CN");
		
		if (system((string("rm -rf ") + directory + ((directory.at(directory.size() - 1) == '/') ? string("") : string("/")) + string("ClassPatientFiles")).c_str()));
		
		if (system((string("mkdir -p ") + directory + ((directory.at(directory.size() - 1) == '/') ? string("") : string("/")) + string("ClassPatientFiles")).c_str()));
		
		for (vector<string>::const_iterator it = classes.begin(); it != classes.end(); ++it)
		{
			xmlFiles.clear();
			
			if (system((string("ls ") + directory + ((directory.at(directory.size() - 1) == '/') ? string("") : string("/")) + *it + string("/*.xml > .temp")).c_str()));
			
			file.open(".temp");
			
			while (file.good())
			{
				if (file.eof()) break;
				
				file.getline(buffer,65536);
				
				if (strlen(buffer) > 0) xmlFiles.push_back(buffer);
			}
			
			file.close();
			
			if (system("rm -rf .temp"));
			
			counter = 0;
			
			INFO("Generating matrices for class: ");
			WARN(*it << endl);
			
			for (vector<string>::const_iterator it2 = xmlFiles.begin(); it2 != xmlFiles.end(); ++it2, ++counter)
			{
				if ((counter % (xmlFiles.size() / 10)) == 0)
				{
					if (ceil(counter + (xmlFiles.size() / 10)) > xmlFiles.size()) ERR("100% done." << endl)
					else ERR(ceil(counter * 100.0 / xmlFiles.size()) << "% done." << endl)
				}
				
				if (system((string("cat ") + *it2 + string(" | grep dateAcquired > .temp")).c_str()));
				
				file.open(".temp");
				
				while (file.good())
				{
					if (file.eof()) break;
					
					file.getline(buffer,65536);
					
					temp = buffer;
					
					break;
				}
				
				file.close();
				
				if (system("rm -rf .temp"));
				
				date = (temp.substr(temp.find(">") + 1)).substr(0,(temp.substr(temp.find(">") + 1)).rfind("<"));
				
				replace(date.begin(),date.end(),' ','_');
				replace(date.begin(),date.end(),':','_');
				
				if (system((string("cat ") + *it2 + string(" | grep visitIdentifier > .temp")).c_str()));
				
				file.open(".temp");
				
				while (file.good())
				{
					if (file.eof()) break;
					
					file.getline(buffer,65536);
					
					temp = buffer;
					
					break;
				}
				
				file.close();
				
				if (system("rm -rf .temp"));
				
				period = (temp.substr(temp.find(">") + 1)).substr(0,(temp.substr(temp.find(">") + 1)).rfind("<"));
				
				if (period.find("Month") == string::npos) continue;
				
				period = period.substr(period.rfind(" ") + 1);
				
				temp = it2->substr(it2->rfind("/") + 1);
				temp = temp.substr(temp.find("_") + 1);
				temp = temp.substr(0,temp.rfind("_"));
				
				string::size_type i = temp.find("_");
				
				for (int j = 0; (j < 2) && (i != string::npos); ++j) i = temp.find("_",i + 1);
				
				temp2 = temp.substr(0,i) + "/" + temp.substr(i+1);
				
				for (vector<string>::const_iterator it3 = sections.begin(); it3 != sections.end(); ++it3)
				{
					temp = directory + ((directory.at(directory.size() - 1) == '/') ? "" : "/") + *it + "/" + temp2.substr(0,temp2.rfind("_")) + "/" + date + "/" + temp2.substr(temp2.rfind("_") + 1) +
						   "/features/descriptors/descriptor_section_" + *it3 + string(".png");
					
					const Mat& image = imread(temp);
					
					if (image.rows < maxFeatureNumber) continue;
					
					file.open((temp.substr(0,temp.rfind("png")) + string("csv")).c_str());
					
					while (file.good())
					{
						if (file.eof()) break;
						
						file.getline(buffer,65536);
						
						temp = buffer;
						
						break;
					}
					
					file.close();
					
					patient = (temp2.substr(0,temp2.rfind("_"))).substr(0,temp2.substr(0,temp2.rfind("_")).rfind("/"));
					
					patientsFile.open((directory + ((directory.at(directory.size() - 1) == '/') ? "" : "/") + string("ClassPatientFiles/") + *it + string("_") + period + string("_section_") + *it3 +
									   string(".csv")).c_str(),ios_base::app);
					
					patientsFile << patient << "," << temp << "," << *it << endl;
					
					patientsFile.close();
				}
			}
		}
		
		for (vector<string>::const_iterator it = classes.begin(); it != classes.end(); ++it)
		{
			csvFiles.clear();
			
			if (system((string("find ") + directory + ((directory.at(directory.size() - 1) == '/') ? string("") : string("/")) + string("ClassPatientFiles/") + string(" -mindepth 1 -maxdepth 1 -name ") + *it +
						string("_*csv > .temp")).c_str()));
			
			file.open(".temp");
			
			while (file.good())
			{
				if (file.eof()) break;
				
				file.getline(buffer,65536);
				
				if (strlen(buffer) > 0) csvFiles.push_back(buffer);
			}
			
			file.close();
			
			if (system("rm -rf .temp"));
			
			for (vector<string>::const_iterator it2 = csvFiles.begin(); it2 != csvFiles.end(); ++it2)
			{
				patientsFile.open((*it2 + string(".new")).c_str());
				
				file.open((*it2).c_str());
				
				oldPatient = "";
				patient = "";
				
				while (file.good())
				{
					if (file.eof()) break;
					
					file.getline(buffer,65536);
					
					if (strlen(buffer) == 0) break;
					
					patient = buffer;
					
					if (oldPatient != patient)
					{
						oldPatient = patient;
						
						patientsFile << patient << endl;
					}
				}
				
				file.close();
				
				patientsFile.close();
				
				if (system((string("rm -rf ") + *it2).c_str()));
				if (system((string("mv ") + *it2 + string(".new ") + *it2).c_str()));
			}
		}
	}
}
