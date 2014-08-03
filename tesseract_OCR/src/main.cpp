#include "stdio.h"
#include "math.h"
#include "tesseract/baseapi.h"
#include "iostream"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "houseNumDetector.h"
#include <sstream>;
#include "testFunctions.h"
using namespace cv;
using namespace std;


int main(){

	char file[255];
	char pathToTest[] = "pointGreyImage";
//	char pathToTest[] = "digit_houseNumber";
	//char pathToTest[] = "jpg4";
	//	extractBinaryTemplate();

//	VideoCapture cap(1); // open the default camera
//	if(!cap.isOpened())  // check if we succeeded
//		return -1;

	for(int imgID = 1; imgID < 4; ++imgID)
//	while(1)
	{
		vector<vector<Point> > contours;
		Mat SrcImage;
		/// Read Image
		sprintf(file, "%s/%d.jpg", pathToTest, imgID);
		SrcImage = imread(file);
//		cap >> SrcImage; // get a new frame from camera
		if (SrcImage.empty())
		{
			cout << "image is empty " << endl;
			continue;
		}
		//imshow("Original  Image", SrcImage); waitKey(1);
		int houseNumber = 0;
		bool flag = 0;
		houseNumDetector myDetector;


		 myDetector.run_main(SrcImage,houseNumber,flag);
//		imshow("Histogram of image",  histImg) ; waitKey(1);
		waitKey(0);
		// if(waitKey(30) >= 0) break;
		// cvDestroyAllWindows();
	}

	/*
    //  Extract the gray Lcd segement using logical operation AND:
    // lcdSergmentOnly &=GrayImage;
    // Pass the binarize gray scale image to tesseract for character recognition
    TesseractOCRResult(DigitsOnly); */

	// distroy all Windows
	waitKey(0);
	cvDestroyAllWindows();
}



