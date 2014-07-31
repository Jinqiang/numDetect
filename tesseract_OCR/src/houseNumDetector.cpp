/*
 * houseNumDetector.cpp
 *
 *  Created on: 29 Jul, 2014
 *      Author: jin
 */

#include "houseNumDetector.h"

houseNumDetector::houseNumDetector() {
	// TODO Auto-generated constructor stub

}

houseNumDetector::~houseNumDetector() {
	// TODO Auto-generated destructor stub
}


void houseNumDetector::cvRotate(Mat& src, double angle, Mat& dst)
{
	int len = std::max(src.cols, src.rows);
	Point2f pt(len/2., len/2.);
	Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);

	warpAffine(src, dst, r, cv::Size(len, len));
}

void houseNumDetector::cvRotatePI(Mat& src, Mat& dst)
{
	//	int len = std::max(src.cols, src.rows);
	Point2f pt(src.cols/2., src.rows/2.);
	Mat r = cv::getRotationMatrix2D(pt, 180, 1.0);

	warpAffine(src, dst, r, cv::Size(src.cols, src.cols));
}

Point houseNumDetector::MatchingMethod( Mat& img, int& match_method, Mat& resultBestMatch, Mat& img_display, int& PatchNum, int& Digitclass, double& matchVal, Rect roiRegion)
{
	Mat result[TempNum][ScalNum];
	int result_cols, result_rows;
	Mat templ[TempNum][ScalNum];
	char file[255];

	double minVal[TempNum][ScalNum];
	double maxVal[TempNum][ScalNum];
	Point minLoc[TempNum][ScalNum];
	Point maxLoc[TempNum][ScalNum];
	Point matchLoc;

	bool MAX = true;
	bool MIN = false;

	int DigitClassMax = 0;
	int DigitClassMin = 0;
	int Patchnum = 0;
	int Patchnummax = 0;
	int Patchnummin = 0;

	/// Source image to display
	img.copyTo( img_display );

	for (int i = 0; i < TempNum; i++)
	{
		// cout << " temp " << i << endl;
		sprintf(file, "%s/%d.jpg", pathToTempl, i);
		Mat templ_raw = imread(file, CV_LOAD_IMAGE_GRAYSCALE);										// Here the image returns a 3-channel color image
		if (!templ_raw.data)
		{
			cout << "File " << file << " not found\n";
			exit(1);
		}

		// imshow("template raw", templ_raw); waitKey(0);
		//		double sizescale_x = TempWidth/templ_raw.cols;
		//		double sizescale_y = TempHeight/templ_raw.rows;
		double xScale = (double)roiRegion.width/templ_raw.cols;
		double yScale = (double)roiRegion.height/templ_raw.rows;
		double scaleFactor = xScale < yScale ? xScale:yScale;
		// cout << "scaleFactor" << scaleFactor << endl;	// double xScale = num1 > num2 ? num1 : num2;
		for (int k = 0; k < ScalNum; k++)
		{
			resize(templ_raw, templ[i][k], Size(), 0.9* scaleFactor, 0.9*scaleFactor, INTER_LINEAR);
			// cout<< "resized image size " << templ[i][k].cols << " X "<< templ[i][k].rows << endl;
			//imshow("template used ", templ[i][k]); waitKey(0);
			result_cols = img.cols - templ[i][k].cols + 1;
			result_rows = img.rows - templ[i][k].rows + 1;

			result[i][k].create( result_cols, result_rows, CV_32FC1 );
			matchTemplate( img, templ[i][k], result[i][k], match_method );

			minMaxLoc( result[i][k], &minVal[i][k], &maxVal[i][k], &minLoc[i][k], &maxLoc[i][k], Mat() );
//			scaleFactor = scaleFactor + ScalStep;
		}
	}

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
	{
		MinMaxSearch(minVal, MIN, DigitClassMax, DigitClassMin, Patchnummax, Patchnummin, matchVal);
		Patchnum = Patchnummin;
		Digitclass = DigitClassMin;
		matchLoc = minLoc[Digitclass][Patchnum];
	}
	else
	{
		MinMaxSearch(maxVal, MAX, DigitClassMax, DigitClassMin, Patchnummax, Patchnummin, matchVal);
		Patchnum = Patchnummax;
		Digitclass = DigitClassMax;
		matchLoc = maxLoc[Digitclass][Patchnum];
	}
	resultBestMatch = result[Digitclass][Patchnum];
	//cout <<"detected number " <<  Digitclass << endl;
	/// Show me what you got
	// circle(img_display, matchLoc, 10, Scalar(255, 255, 255), 3);
	// rectangle( img_display, matchLoc, Point( matchLoc.x + templ[Digitclass][Patchnum].cols , matchLoc.y + templ[Digitclass][Patchnum].rows ), Scalar::all(255), 2, 8, 0 );
	// rectangle( resultBestMatch, matchLoc, Point( matchLoc.x + templ[Digitclass][Patchnum].cols , matchLoc.y + templ[Digitclass][Patchnum].rows ), Scalar::all(255), 2, 8, 0 );
	// imshow("Template used" , templ[Digitclass][Patchnum]); waitKey(0);
	return matchLoc;
}

void houseNumDetector::MinMaxSearch( double (*ArrayVal)[ScalNum], bool flag, int& DigitClassMax, int& DigitClassMin, int& Patchnummax, int& Patchnummin, double& matchVal)
{
	double mmax = ArrayVal[0][0];
	double mmin = ArrayVal[0][0];

	for (int i = 0; i < TempNum; i++)
	{
		for (int j = 0; j < ScalNum; j++)
		{
			if(ArrayVal[i][j] > mmax)
			{
				mmax = ArrayVal[i][j];
				DigitClassMax = i;
				Patchnummax = j;
			}
			else if(ArrayVal[i][j] < mmin)
			{
				mmin = ArrayVal[i][j];
				DigitClassMin = i;
				Patchnummin = j;
			}

		}
	}

	if(flag)
		matchVal = mmax;
	else
		matchVal = mmin;
}

void houseNumDetector::thresholdImage(Mat img_gray, Mat& output)
{
	/// show equalized histogram image
	//		  Mat histoEqualizedImage;
	//    	  equalizeHist(img_gray, histoEqualizedImage);
	//        imshow("equalized histograme image", histoEqualizedImage); waitKey(0);

	// 1 hard threshold;Not Used
//		int thresh      = 150; //image intensity threshold
//		threshold(img_gray,output,thresh,255,CV_THRESH_BINARY);
	//	imshow(" Current Image threshold",img_thres);
//	/// 2 adaptive threshold
	Mat adThresholdOutput;
	adaptiveThreshold(img_gray,adThresholdOutput,255,
			CV_ADAPTIVE_THRESH_MEAN_C,
			CV_THRESH_BINARY,
			75,
			10);
//	//  imshow(" Adaptive thresholding image",adThresholdOutput);     waitKey(0);
//	/// 3 binary thresholding using OTSU method
	Mat img_thrOTSU;
	threshold(img_gray, img_thrOTSU, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//  imshow("image thresh OTSU", img_thrOTSU); waitKey(0);

	// OR operation of the above two thresholding method.
	 Mat orImage;
	max(img_thrOTSU, adThresholdOutput, output);

	// imshow("OR image operation", output); waitKey(0);
	// output = img_thrOTSU;
}

void houseNumDetector::getBinaryImage(Mat img_gray, Mat& Output)
{
	/// remove noise
	// 1. pyromid
	//  Mat pyr, timg;
	//	pyrDown(img_gray, pyr, Size(SrcImage.cols/2, SrcImage.rows/2));
	//	pyrUp(pyr, timg, img_gray.size());
	//	imshow(" pyro  blur Image ",timg);
    // 2. blur
	// blur(img_gray, img_gray, Size(5,5));
	GaussianBlur(img_gray,img_gray,Size(3,3),0);
	// imshow(" Gaussian blur Image ",img_gray);     waitKey(0);

	Mat img_threshold= Mat::zeros(img_gray.size(), img_gray.type());

	/// threshold type 1,2,3
	thresholdImage( img_gray, img_threshold);

	/// threshold type 4, OTSU in multiple ROIs
	 //thresholdImageInROIs(img_gray, img_threshold);

	// morphological operation
	Mat dst;
	Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
	morphologyEx(img_threshold, img_threshold, CV_MOP_OPEN, element);

	//  morphologyEx(dst, dst, CV_MOP_OPEN, element);

	/// biwise_not to avoid contouring on the edge of image
	bitwise_not(img_threshold, Output);
//	imshow("Binary image" , Output); waitKey(1);
}

void houseNumDetector::removeContourWithLargerAngle(Mat inputToContourSelection)
{
	///find external contour and remove contours with larger rotated angle;
	vector<vector<Point> > contours;
	findContours( inputToContourSelection.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
	// displayContours(inputToContourSelection, contours);
	for (int i = 0; i < contours.size(); i++) {
		RotatedRect minRect;
		minRect = minAreaRect( Mat(contours[i]) );
		double diff_angle = abs(minRect.angle + 45);  // compared to -45 deg, if in -15 ~ 75 deg, remove the contour
		if (diff_angle < 20) //
		{
			cout << " Removed due to larger angle" << diff_angle << endl;
			cv::drawContours(inputToContourSelection,
					contours, i, //Scalar(255,255,0),
					Scalar::all(0),
					CV_FILLED);
			//imshow(" Thresholded Image",inputOutput);  // waitKey(0);
		}
	}
}

void houseNumDetector::thresholdImageInROIs(Mat img_gray, Mat& output)
{
	int N = 300;
	for (int r = 0; r < img_gray.rows; r += N/2)
		for (int c = 0; c < img_gray.cols; c += N/2)
		{
			cv::Mat tile = img_gray(Range(r, MIN(r + N, img_gray.rows)),Range(c, MIN(c + N, img_gray.cols) ) ); //no data copying here
			//cv::Mat tileCopy = img(cv::Range(r, min(r + N, img.rows)),
			//cv::Range(c, min(c + N, img.cols))).clone();//with data copying
			//			cout << "size " << tile.size();
			//			imshow(" local Image ",tile);
			Mat temp;
			threshold(tile, temp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			//			imshow(" local Image binary",temp);
			Rect bRect;
			bRect.x = c;
			bRect.y = r;
			bRect.height = MIN(r + N, img_gray.rows) - r ;
			bRect.width  = MIN(c + N, img_gray.cols) - c ;
			output(bRect) = output(bRect)|temp;
			//				temp.copyTo(roi_thrOTSU(bRect));
			// max(temp, roi_thrOTSU(bRect), roi_thrOTSU(bRect) );
			// imshow(" global Image binary",output);     waitKey(1);
			//tile can be smaller than NxN if image size is not a factor of N
		}

	//thresholdImage(img_gray, orImage);
	//	imshow(" Binary final ",roi_thrOTSU);
}

// This function take in a Gray Scale image finds Contours of the Image.
// Loops through all the contours and only draws contours bellow a specified area size.
// and returns the processed image
void houseNumDetector::imageContourThresholdingHierarchy(Mat& inputOutput)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat input = inputOutput.clone();
	//cv::findContours(input, contours,  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	findContours( input, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	// displayContours(input, contours);
	Mat drawing = Mat::zeros(input.size(), CV_8UC3);

	double epsilon = 5 ; // 5 pixel accuracy for approxPolyDP(Mat(contours[i]),contourDP[i], epsilon, true);
	for (int i = 0; i < contours.size(); i++) {
		Rect bRect;
		bRect   = boundingRect(Mat(contours[i]) );
		//		// #ifdef DEBUGff
		//		//		cout << endl << i <<  " in "  << contours.size() << endl;
		//		//		cout << "hierarchy idx: " << hierarchy[i] << endl;
		//		//		cout << contours[i]<< endl;
		//		RotatedRect minRect;
		//		minRect = minAreaRect( Mat(contours[i]) );
		//		cout <<"angle of rect" << minRect.angle << endl;
		//
		//		Point2f rect_points[4]; minRect.points( rect_points );
		//		cout << "rotate vertices " << rect_points[0] << endl;
		//		rectangle( drawing, Point(bRect.x, bRect.y), Point(bRect.x + bRect.width, bRect.y + bRect.height),Scalar( 0, 0, 255 ));
		//		for( int j = 0; j < 4; j++ ){
		//			//cout<< rect_points[j] << endl;
		//			line( drawing, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,0), 1, 8 );
		//		}
		//
		//		//|| !checkContourAspectRatio(contours[i])
		//		vector<Point> contourDP;
		//		approxPolyDP(Mat(contours[i]),contourDP, epsilon, true);
		//
		//		cout << "isContourConvex " << isContourConvex(contours[i]) << "  DP "<<  isContourConvex(contourDP) << endl;
		//		cout << "countour area  " << (contourArea(contours[i]) < 100) << endl;
		//		cout << "countour size  " << (contours[i].size()< 5 )<< endl;
		//		cout << "conour on border " << checkContourCloseBorder(input, bRect) << endl;
		//		cout << "check contour aspect ratio  " << (!checkContourAspectRatio(contours[i])) << endl;
		//  #endif

		if (       fabs(contourArea(contours[i])) < 100  // check area
				|| fabs(contourArea(contours[i])) > inputOutput.rows * inputOutput.cols * 0.25
				|| contours[i].size()    < 5   // check point size
				|| checkContourCloseBorder(input, bRect) // check if in border
				|| !checkContourAspectRatio(contours[i])) // check if in aspect ratio
		{
			cv::drawContours(inputOutput,
					contours, i, //Scalar(255,255,0),
					Scalar::all(0),
					CV_FILLED);
		}
		//		imshow(" Thresholded Image",inputOutput); //waitKey(0);
		//		imshow("Contour drawing filtered", drawing);
	}
}

// This function take RGB image.Then convert it into HSV for easy colour detection
// and threshold it with yellow to green part as white and all other regions as black.
// Then return that image.
Mat houseNumDetector::GetThresholdedHSVImage( cv::Mat TempSourceImage )
{

	cv::Mat HSVImage;
	cv::Mat ThresholdImage;

	// Convert source SrcImage  from RBG to to HSV Image foreasy colour detection
	cvtColor( TempSourceImage, HSVImage, CV_BGR2HSV );

	// Create binary thresholded image  to max/min HSV ranges
	// For detecting yellow to green LCD Area in  the Image - HSV mode

	//inRange( HSVImage, Scalar( 20, 100, 100 ), Scalar( 70,  255, 255 ),ThresholdImage );

	inRange( HSVImage, Scalar(26,93,55), Scalar(45,135,238), ThresholdImage );

	return ThresholdImage;

}

// This function Loops through all the Contours to get the largest
// and returns the index of the Largest Contour.
int houseNumDetector::GetlargestContour(vector<vector<Point> > contours) {

	double MaxArea     = 0;
	int    MaxIndex    = 0;
	int    ContourSize = contours.size();

	for (int i = 0; i < ContourSize; i++) {

		if (MaxArea  < cv::contourArea(contours[i])) {
			MaxArea  = cv::contourArea(contours[i]);
			MaxIndex = i;
		}
	}

	return MaxIndex;

}

bool houseNumDetector::checkContourAspectRatio(vector<Point> contours)
{
	// this function checks the aspect ratio of the contours
	// range defined as (1 -  range) * a4_ratio ~ (1 + range) * a4_ratio
	float a4_ratio = 0.7071; // A4 paper ratio
	float range = 0.4; // acceptable paper scale range

	//	// rotatedRect aspect ratio, not good, do not use
	//	RotatedRect temp_rect = minAreaRect(Mat(contours));
	//	float temp_ratio = temp_rect.size.width / temp_rect.size.height;
	//	cout << "Rotate bounding angle shape ratio: " << temp_ratio << endl;

	/// use rectangle ratio;
	Rect bRect;
	bRect   = boundingRect(Mat(contours) );
	float temp_ratio = (float)bRect.width/bRect.height;
#ifdef DEBUG
	cout << "Bounding box shape ratio " << temp_ratio << endl;
#endif
	bool output;
	output = 0;
	if(temp_ratio > (1 - range)*a4_ratio && temp_ratio < (1 + range)*a4_ratio)
	{
		output = 1;
	}
	return output;
}

void houseNumDetector::getContourROI(Mat input, vector< Rect >& potentialROIs)
{
	//	vector<vector<Point> > contours;
	//	cv::findContours(input.clone(),
	//			contours,
	//			CV_RETR_EXTERNAL,
	//			CV_CHAIN_APPROX_NONE);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat temp = input.clone();
	//cv::findContours(input, contours,  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	findContours( input, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	Mat drawing = Mat::zeros( input.size(), CV_8UC3 );

	for (int i = 0; i < contours.size(); ++i)
	{
		// cout << i << "  contour size "  << contours.size() << endl;
		// cout << contours[i] << endl;
		Rect bRect;
		bRect   = boundingRect(Mat(contours[i]) );
		//		cout << " bounding Rect pos" << bRect.x << "  " << bRect.y << endl;
		if(bRect.area() > 50
				&& checkContourAspectRatio(contours[i]))
		{
			potentialROIs.push_back(bRect);
		}
		rectangle( drawing, Point(bRect.x, bRect.y),
				Point(bRect.x + bRect.width,bRect.y + bRect.height),
				Scalar( 0, 0, 255 ));
	}
	// imshow("List of Potential ROI ", drawing); waitKey(0);
}


void houseNumDetector::getContourROI(Mat input, vector< Rect >& potentialROIs, vector<vector<Point> >& contourOutput)
{
	//	vector<vector<Point> > contours;
	//	cv::findContours(input.clone(),
	//			contours,
	//			CV_RETR_EXTERNAL,
	//			CV_CHAIN_APPROX_NONE);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat temp = input.clone();
	//cv::findContours(input, contours,  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	findContours( input, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	Mat drawing = Mat::zeros( input.size(), CV_8UC3 );

	for (int i = 0; i < contours.size(); ++i)
	{
		// cout << i << "  contour size "  << contours.size() << endl;
		// cout << contours[i] << endl;
		Rect bRect;
		bRect   = boundingRect(Mat(contours[i]) );
		//		cout << " bounding Rect pos" << bRect.x << "  " << bRect.y << endl;
		if(bRect.area() > 50
				&& checkContourAspectRatio(contours[i]))
		{
			// enlarge the ROI by 2 pixels in X, Y Direction
			bRect -= Point(2,2);
			bRect += Size(4,4);
			potentialROIs.push_back(bRect);
			contourOutput.push_back(contours[i]);
		}
		rectangle( drawing, Point(bRect.x, bRect.y),
				Point(bRect.x + bRect.width,bRect.y + bRect.height),
				Scalar( 0, 0, 255 ));
	}
	// imshow("List of Potential ROI ", drawing); waitKey(0);
}


// This function take in Binarized Image passes it to tesseract OCR.
// and. returns the converted text from the image
// all in all converts image to text.

//char* TesseractOCRResult(cv::Mat TempSourceImage) {
//
//	tesseract::TessBaseAPI TessOCR;
//	//Initialize Tesseract to only english training data
//	TessOCR.Init(NULL, Tesseractlanguage );
//
//	// Specifiy the Tesseract whitelist to only look for digits
//	TessOCR.SetVariable(TesseractCharWhitelist, TesseractDigitsOnly);
//
//	TessOCR.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
//
//	// Pass the binarized image to tesseract
//	TessOCR.SetImage((uchar*)TempSourceImage.data,
//			TempSourceImage.cols,
//			TempSourceImage.rows,
//			1,
//			TempSourceImage.cols);
//
//	char* PinNumber = TessOCR.GetUTF8Text();
//	std::cout << PinNumber << std::endl;
//
//	return PinNumber;
//
//}

void houseNumDetector::displayContours(Mat SmallContoursRemoved, vector< vector<Point> > contours)
{
	// Draw contours + rotated rects + ellipses
	Mat drawing = Mat::zeros( SmallContoursRemoved.size(), CV_8UC3 );

	for (int i = 0; i < contours.size(); ++i)
	{
		// cout << i << "  contour size "  << contours.size() << endl;
		// cout << contours[i] << endl;
		RotatedRect minRect;
		Rect bRect;
		minRect = minAreaRect( Mat(contours[i]) );
		bRect   = boundingRect(Mat(contours[i]) );
		Point2f rect_points[4]; minRect.points( rect_points );
		rectangle( drawing, Point(bRect.x, bRect.y), Point(bRect.x + bRect.width, bRect.y + bRect.height),Scalar( 0, 0, 255 ));
		for( int j = 0; j < 4; j++ ){
			cout<< rect_points[j] << endl;
			line( drawing, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,0), 1, 8 );
		}
	}
	// imshow("Drawing of contour 2", drawing); waitKey(1);
}


bool houseNumDetector::checkContourCloseBorder(Mat img, Rect bRect)
{
	int row_lower = 0;
	int row_upper = img.rows;
	int col_lower = 0;
	int col_upper = img.cols;
	int threshold = 5 ;
	bool output = 0;  // default is not on border, 0; if on border = 1;
	// check leftup corner
	if( abs(bRect.x - col_lower) < threshold || abs(bRect.y - row_lower) < threshold){
		output = 1;
		return output;
	}
	// check right-down corner
	if( abs(bRect.x + bRect.width - col_upper) < threshold ||
			abs(bRect.y + bRect.height - row_upper) < threshold)
	{
		output = 1;
		return output;
	}
	return output;
}

void houseNumDetector::run_main(Mat& SrcImage, int& houseNumber, bool& flag)
{
	resize(SrcImage, SrcImage, Size(640,480));
	//		imshow("Original  Image", SrcImage); //waitKey(0);

	/// Detect edges using Threshold
	Mat img_gray;
	cvtColor(SrcImage,img_gray,CV_RGB2GRAY);

	Mat inputToContourSelection;
	getBinaryImage(img_gray, inputToContourSelection);

	/// apply thresholding to the contours
	imageContourThresholdingHierarchy(inputToContourSelection);
	// imshow(" Output Image ",inputToContourSelection);

	Mat input = inputToContourSelection.clone();
	vector< Rect > potentialROIs;
	vector<vector<Point> > contourOutput;
	getContourROI( input, potentialROIs, contourOutput);

	/// not operation back for template matching
	bitwise_not(inputToContourSelection,inputToContourSelection);
	// imshow("Input image for temp Matching", inputToContourSelection); waitKey(1);

	/// run template matching for all ROIs in the list
	Mat img_raw, result, img_display, reresult, reimg_display;
	int match_method = 5;
	int PatchNum     = 0;
	double matchVal  = 0;
	vector< int > digitList;
	vector< double > valueList;
	vector< Point> locList;
	// loop through all extracted ROIs
	if(potentialROIs.size() > 0)
	{
		for (int i = 0 ; i < potentialROIs.size(); ++i)
		{
			//cout << " ROI number --- " << i << endl;
			Mat img;
			Mat(inputToContourSelection, potentialROIs[i]).copyTo(img);

			int PatchNum     = 0;
			int Digitclass   = 0;
			Point MatchLoc = MatchingMethod( img, match_method, result, img_display, PatchNum, Digitclass, matchVal, potentialROIs[i]);
			//cout <<"detected number " <<  Digitclass << endl;
			//cout <<"match value " <<  matchVal << endl;
			valueList.push_back(matchVal);
			digitList.push_back(Digitclass);
			locList.push_back(MatchLoc); //waitKey(0);
		}

		double maxValue = 0;
		int index = 0;
		for(int i =0 ; i<valueList.size(); ++i){
			if(maxValue < valueList[i])
			{
				maxValue = valueList[i] ;
				index = i;
			}
		}
		cerr << "Final Number is " << digitList[index] << "  with MatchVal " << maxValue << endl;
		if (maxValue >0.3){
			stringstream ss;
			ss << "Detected number is : " ;
			ss << digitList[index];
			ss << "  Match value: " ;
			ss << maxValue ;
			string str = ss.str();

			putText( SrcImage, str, Point(10,30), FONT_HERSHEY_DUPLEX, 0.7,
					Scalar(0, 0, 255), 0.5, 8 );
			Rect bRect;
			bRect.x = potentialROIs[index].x;
			bRect.y = potentialROIs[index].y;
			bRect.width = potentialROIs[index].width;
			bRect.height =  potentialROIs[index].height;
			rectangle( SrcImage, Point(bRect.x, bRect.y), Point(bRect.x + bRect.width, bRect.y + bRect.height), Scalar( 0, 0, 255 ));
			// plot rotate rect
			RotatedRect minRect;
			minRect = minAreaRect( Mat(contourOutput[index]) );
			// cout << "angle of contour " << minRect.angle << endl;
			Point2f rect_points[4]; minRect.points( rect_points );
			for( int j = 0; j < 4; j++ ){
				line( SrcImage, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,0), 1, 8 );
			}
			imshow("input ROI", Mat(inputToContourSelection, potentialROIs[index]).clone());
			cout << " input ROI " << potentialROIs[index].width << endl;
			houseNumber = digitList[index];
			flag = 1;
		}
		else{
			stringstream ss;
			ss << " No number detected !!! " ;
			string str = ss.str();
			putText( SrcImage, str, Point(10,30), FONT_HERSHEY_DUPLEX, 0.7,
					Scalar(0, 255, 255), 0.5, 8 );
			flag = 0; //  no number detected;
		}
	}
	else{
		stringstream ss;
		ss << " Not enough ROIs!!! " ;
		string str = ss.str();
		putText( SrcImage, str, Point(10,30), FONT_HERSHEY_DUPLEX, 0.7,
				Scalar(255, 255), 0.5, 8 );
	}
//		resize(SrcImage, SrcImage, Size(), 0.5, 0.5, INTER_LINEAR);
	imshow("Source with number" , SrcImage);
}


void houseNumDetector::minizeTemplate()
{
	char file[255];
	char pathToTempl[] = "Training_housenumber";
	char pathToSmallTempl[] = "Training_housenumber_Crop";
	for (int i = 0; i < 10; ++i)
	{
		// cout << " temp " << i << endl;
		sprintf(file, "%s/%d.jpg", pathToTempl, i);
		Mat templ_raw = imread(file, CV_LOAD_IMAGE_GRAYSCALE);										// Here the image returns a 3-channel color image
		if (!templ_raw.data)
		{
			cout << "File " << file << " not found\n";
			exit(1);
		}
		resize(templ_raw, templ_raw, Size(), 0.1, 0.1,  INTER_LINEAR);
		cout << "image size" << templ_raw.size() << endl;
		sprintf(file, "%s/%d.jpg", pathToSmallTempl, i);
		imwrite(file, templ_raw );
		imshow("template read", templ_raw); waitKey(0);
	}
}


void houseNumDetector::extractBinaryTemplate()
{
	char file[255];
	char pathToTempl[] = "Training_housenumber";
	char pathToSmallTempl[] = "Training_housenumber_Binary";
	for (int i = 0; i < 10; ++i)
	{
		// cout << " temp " << i << endl;
		sprintf(file, "%s/%d.jpg", pathToTempl, i);
		Mat templ_raw = imread(file, CV_LOAD_IMAGE_GRAYSCALE);										// Here the image returns a 3-channel color image
		if (!templ_raw.data)
		{
			cout << "File " << file << " not found\n";
			exit(1);
		}
		resize(templ_raw, templ_raw, Size(), 0.3, 0.3,  INTER_LINEAR);
		Mat binaryImage;
		threshold(templ_raw, binaryImage, 100,255, CV_THRESH_BINARY);
		Mat binary_INV;
		bitwise_not(binaryImage, binary_INV);
		vector<vector<Point> > contours;
		findContours( binary_INV, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
		cout << "contour size"  << contours.size() << endl;
		displayContours(binaryImage, contours);
		Rect bRect;
		bRect   = boundingRect(Mat(contours[0]) );
		bRect -=Point(2,2);
		bRect +=Size(4,4);
		Mat ROI;
		Mat(binaryImage, bRect).copyTo(ROI);
		cout << "image size" << templ_raw.size() << endl;
		sprintf(file, "%s/%d.jpg", pathToSmallTempl, i);
		imwrite(file, ROI );
//		sprintf(file, "%s/%d.jpg", pathToSmallTempl, i+10);
//		imwrite(file, binaryImage );
		imshow("template binaryImage", binaryImage);
		imshow("template binaryImage", ROI); waitKey(0);
	}
}

void houseNumDetector::saveImageToLocal(Mat input, int ID)
{
	char file[255];
	char pathToImage[] = "InputImage";									// Here the image returns a 3-channel color image
		if (!input.data)
		{
			cout << "Image is empty " << endl;
			return;
		}
		sprintf(file, "%s/%d.jpg", pathToImage, ID);
		imwrite(file, input );
		imshow("template read", input); waitKey(1);
}
