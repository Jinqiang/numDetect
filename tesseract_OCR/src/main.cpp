#include "stdio.h"
#include "math.h"
#include "tesseract/baseapi.h"
#include "iostream"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "houseNumDetector.h"
#include <sstream>;
using namespace cv;
using namespace std;


int main(){

	char file[255];
//	char pathToTest[] = "digit_houseNumber";
//	char pathToTest[] = "jpg4";
	//	extractBinaryTemplate();

		VideoCapture cap(1); // open the default camera
		if(!cap.isOpened())  // check if we succeeded
			return -1;

	// for(int imgID = 1; imgID < 22; ++imgID)
	//	namedWindow("Source with number", 1);
	//	moveWindow("Source with number", 10, 10);
	//	namedWindow("Input image for temp Matching",1);
	//	moveWindow("Input image for temp Matching", 600, 10);
//	for(int imgID = 1000; imgID < 1131; ++imgID)
			 while(1)
	{

		/// Read Image
		vector<vector<Point> > contours;
//		sprintf(file, "%s/%d.jpg", pathToTest, imgID);
//		//Mat SrcImage = imread("digit_houseNumber/12.bmp");
//		Mat SrcImage = imread(file);
				Mat SrcImage;
				cap >> SrcImage; // get a new frame from camera
		if (SrcImage.empty())
		{
			cout << "image is empty " << endl;
			continue;
		}
		//imshow("Original  Image", SrcImage); waitKey(1);
		// cout << SrcImage.size() << endl;
		int houseNumber = 0;
		bool flag = 0;
		houseNumDetector myDetector;
		myDetector.run_main(SrcImage,houseNumber,flag);
		waitKey(10);
		// if(waitKey(30) >= 0) break;
		// cvDestroyAllWindows();
	}

	/*
    //  Extract the gray Lcd segement using logical operation AND:
    // lcdSergmentOnly &=GrayImage;
    // Pass the binarize gray scale image to tesseract for character recognition
    TesseractOCRResult(DigitsOnly); */

	// distroy all Windows
	waitKey(500);
	cvDestroyAllWindows();
}



