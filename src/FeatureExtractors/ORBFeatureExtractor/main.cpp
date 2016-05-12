#include "ORBFeatureExtractor.h"
#include <Utils/DebugUtils.h>

using namespace std;

static const string USAGE = "Usage: ./FeatureExtractor -d <directory-root>.";

int main(int argc, char** argv)
{
	if ((argc < 3) || (argc > 3) || ((argc > 1) && (strcmp(argv[1],"-d") != 0)))
	{
		ERR(USAGE << endl);
		
		exit(-1);
	}
	
	FeatureExtractors::ORBFeatureExtractor featureExtractor;
	
	featureExtractor.exec(argv[2]);
	
	return 0;
}
