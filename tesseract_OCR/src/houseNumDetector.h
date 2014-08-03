/*
 * houseNumDetector.h
 *
 *  Created on: 29 Jul, 2014
 *      Author: jin
 */

#ifndef HOUSENUMDETECTOR_H_
#define HOUSENUMDETECTOR_H_

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <string.h>

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

using namespace cv;
using namespace std;

const char pathToImages[] = "digit_houseNumber/";
const char pathToTempl[] = "Training_housenumber_Binary";//"Training_housenumber";//"//
const size_t TempNum = 10;
const size_t ScalNum = 1;

class houseNumDetector {
private:
	bool _displayFlag;

public:
	houseNumDetector();
	virtual ~houseNumDetector();

public:
	Point MatchingMethod(Mat& img, int& match_method, Mat& resultBestMatch, Mat& img_display, int& PatchNum, int& Digitclass, double& matchVal, Rect roi);
	void MinMaxSearch( double (*ArrayVal)[ScalNum], bool flag, int& DigitClassMax, int& DigitClassMin, int& Patchnummax, int& Patchnummin, double& matchVal);
	void cvRotate(Mat& src, double angle, Mat& dst);
	void cvRotatePI(Mat& src, Mat& dst);

	void run_main(Mat& SrcImage, int& houseNumber, bool& flag);
	void thresholdImage(Mat img_gray, Mat& output);
	void getBinaryImage(Mat img_gray, Mat& Output);
	void thresholdImageInROIs(Mat img_gray, Mat& output);
	void imageContourThresholdingHierarchy(Mat& inputOutput);
	void imageContourThresholdROI(Mat& inputOutput, vector<Rect>& potentialROIs, vector<vector<Point> >& contours);

	bool checkContourAspectRatio(vector<Point> contours);
	bool checkContourCloseBorder(Mat img, Rect bRect);
	void removeContourWithLargerAngle(Mat inputToContourSelection);
	void displayContours(Mat SmallContoursRemoved, vector< vector<Point> > contours);
	Mat GetThresholdedHSVImage( cv::Mat TempSourceImage );
	int GetlargestContour(vector<vector<Point> > contours);
	void getContourROI(Mat input, vector< Rect >& potentialROIs, vector<vector<Point> >& contourOutput);
	void getContourROI(Mat input, vector< Rect >& potentialROIs);
	void extractBinaryTemplate();
	void minizeTemplate();
	void saveImageToLocal(Mat input, int ID);
	Mat  showHistogram(const cv::Mat inImage);
	/*
	const char* Tesseractlanguage = "eng"; // initialize Tesseract with english language training data
	const char* TesseractCharWhitelist = "tessedit_char_whitelist"; // limit tesseract to recognize only the wanted characters
	const char* TesseractDigitsOnly = "0123456789"; // Recognise Digits only

	char* TesseractOCRResult(cv::Mat TempSourceImage); */
};

#endif /* HOUSENUMDETECTOR_H_ */
