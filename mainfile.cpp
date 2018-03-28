#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4
  #include <opencv2/nonfree/features2d.hpp>
#endif

#include <iostream>
#include <stdio.h>

#include <sys/time.h>
#include <cstdlib>
#include <math.h>

#include <sstream>
#include <iomanip>

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

using namespace std;
using namespace cv;

struct Motorbike{
	vector<Point> vec;
	timeval start;
	Rect roi;
	Point center;
	int id;
};

int curIndex = 1;

/** Function Headers */
void detectAndDisplay(Mat frame);
string toleadstr(int val, int length);

/** Global variables */
string root = "/media/DATA/OPENCV/";
string videoFolder = root + "videos/";
string dataFolder = root + "data/";
string exportImgFolder = root + "/imgs/";

String motor_cascade_name = dataFolder + "motor-v4.xml";
CascadeClassifier motor_cascade;
string window_name = "Sac Le - Motorbike detection and counting";
Rect roi(391, 303, 725, 415 - 30);
vector<Motorbike > listMotors;
Rect firstFrameCondition(Point(433,322), Point(662,349));

/** Create some points */
Point corners[1][4];
const Point* corner_list[1] = { corners[0] };

int num_points = 4;
int num_polygons = 1;
int line_type = 8;

int total = 0;

int fileIndx = 0;

VideoWriter outputVideo;

void drawPolyline(Mat frame) {
	vector<Point> contour;
	contour.push_back(Point(391 + 42, 320)); //1
	contour.push_back(Point(391, 303 + 380)); //2
	contour.push_back(Point(frame.cols - 290, frame.rows - 5)); //3
	contour.push_back(Point(391 + 338 - 120, 303)); //4


	const cv::Point *pts = (const cv::Point*) Mat(contour).data;
	int npts = Mat(contour).rows;
	// draw the polygon

	polylines(frame, &pts, &npts, 1, true, // draw closed contour (i.e. joint end to start)
			Scalar(0, 255, 0),// colour RGB ordering (here = green)
			2, // line thickness
			CV_AA, 0);
}

/** @function main */
int main(int argc, const char** argv) {
	Mat frame;

	string url = dataFolder + "DNG13_20180328148.mp4";

	if(argc > 1){
		cout<<"Loading file : "<<argv[1]<<endl;
		url = dataFolder + argv[1];
	}

	VideoCapture inputVideo(url);

	const string NAME = videoFolder + "export_ex.avi"; // Form the new name with container

	//-- 1. Load the cascades
	if (!motor_cascade.load(motor_cascade_name)) {
		printf("--(!)Error loading\n");
		return -1;
	};

	inputVideo >> frame;
	Rect roix(0, 0, frame.cols, frame.rows - 30);
	frame = frame(roix);

	Size S = Size((int) frame.cols, // Acquire input size
			(int) frame.rows);

	outputVideo.open(NAME, CV_FOURCC('M', 'J', 'P', 'G'),
			24, S, true);

	if (!outputVideo.isOpened()) {
		cout << "Could not open the output video for write: " << NAME << endl;
		return -1;
	}

	corners[0][0] = Point(42, 0);
	corners[0][1] = Point(0, 385);
	corners[0][2] = Point(597, 385);
	corners[0][3] = Point(214, 0);


	vector<KeyPoint> keypoints;

	if (inputVideo.isOpened()) {
		while (true) {
			inputVideo >> frame;
			frame = frame(roix);
			//-- 3. Apply the classifier to the frame
			if (!frame.empty()) {
				detectAndDisplay(frame);
			} else {
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			int c = waitKey(1);
			if ((char) c == 'c') {
				break;
			}
		}
	}
	return 0;
}

int sizeA(int arr[]){
	return sizeof(arr)/sizeof(arr[0]);
}

void drawVectors(Mat mat, int id){
	timeval now;
	gettimeofday(&now, NULL);

	int ids[100];

	int indx = 0;
	int motorSize = listMotors.size();

	//Remove motor not run
	for(int i=0; i<motorSize; i++){
		Motorbike currMotor = listMotors[i];
		//Calculator delta time by second
		float delta = ((now.tv_sec  - currMotor.start.tv_sec) * 1000000u +
						now.tv_usec - currMotor.start.tv_usec) / 1.e6;
		//If delta 2 second will be erase this motor
		if(delta >= 2){
			ids[indx] = currMotor.id;
			indx++;
		}
	}

	for(int j=motorSize-1; j>=0; j--){
		Motorbike currMotor = listMotors[j];
		for(int h=0; h<sizeA(ids); h++){
			if(currMotor.id == ids[h]){
				listMotors.erase(listMotors.begin() + j);
				break;
			}
		}
	}

	for(int i=0; i<listMotors.size(); i++){
		//Check if current ID will put text and id
		if(listMotors[i].id == id){
			vector<Point> points = listMotors[i].vec;
			//do something here
			string fileName = "img_" + toleadstr(id, 3) + "_" + toleadstr(fileIndx, 5) + ".jpg";
			string filePath = exportImgFolder + fileName;
			fileIndx++;

			imwrite(filePath, mat(listMotors[i].roi));

			putText(mat, SSTR(id), listMotors[i].center, 3, 0.7, Scalar(255, 255,255), 1,1,false);

			for(int j = 0; j<points.size(); j++){
				if(points.size() > 2 && j<points.size()-1){
					line(mat, points[j], points[j+1], Scalar(5,201,15), 2, 8, 0);
				}
			}
		}
	}
}
/**
*
*
*
 */
void detectAndDisplay(Mat frame) {
	std::vector<Rect> motors;
	Mat frame_gray, processFrame;
	Mat cut = frame(roi);

	Mat mask(cut.rows, cut.cols, CV_8UC3, Scalar(0, 0, 0));
	fillPoly(mask, corner_list, &num_points, num_polygons,
			Scalar(255, 255, 255), line_type);
	bitwise_and(cut, mask, processFrame);
	//processFrame = cut;
	cvtColor(processFrame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect motorbike
	motor_cascade.detectMultiScale(frame_gray, motors, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(23, 50), Size(50, 108));

	// loop foreach motor
	for (size_t i = 0; i < motors.size(); i++) {
		Point center(motors[i].x + 391 + motors[i].width / 2, motors[i].y + 303 + motors[i].height/2);

		Rect findRoi(motors[i].x + 391, motors[i].y + 303, motors[i].width, motors[i].height);


		Mat faceROI = frame_gray(motors[i]);

		// add to list if list is empty
		if(listMotors.empty()){
			vector<Point> line;
			Motorbike motor;
			line.push_back(center);
			motor.vec = line;
			motor.id = curIndex;
			motor.roi = findRoi;
			motor.center = center;
			if(curIndex > total){
				total++;
			}
			gettimeofday(&motor.start, NULL);
			curIndex++;
			listMotors.push_back(motor);
			drawVectors(frame, motor.id);
		}else{
			bool hasBefore = false;
			//Check current motor is exists on list
			for(int xx =0 ; xx < listMotors.size(); xx++){
				vector<Point> line = listMotors[xx].vec;
				if(!line.empty()){
					Point beforePoint = line[line.size() - 1];
					int deltaX = abs(center.x - beforePoint.x);
					int deltaY = abs(center.y - beforePoint.y);
					if(deltaX < 20 && deltaY < 20){
						hasBefore = true;
						//Cap nhat lai center cua xe may
						listMotors[xx].vec.push_back(center);
						listMotors[xx].roi = findRoi;
						listMotors[xx].center = center;
						drawVectors(frame, listMotors[xx].id);
						break;
					}
				}
			}

			//Truong hop khong ton tai xe truoc do trong list, thuc hien them xe moi
			if(!hasBefore && firstFrameCondition.contains(center)){
				vector<Point> line;
				line.push_back(center);
				Motorbike motor;
				motor.vec = line;
				motor.roi = findRoi;
				motor.center = center;
				motor.id = curIndex; curIndex++;
				if(curIndex > total){
					total++;
				}
				gettimeofday(&motor.start, NULL);
				listMotors.push_back(motor);
				drawVectors(frame, motor.id);
			}
		}


	}

	drawPolyline(frame);

	//draw white matrix
	//150x300
	Mat backg( 100, 250, CV_8UC3, Scalar(255,255,255));
	backg.copyTo(frame.rowRange(10, backg.rows+10).colRange(frame.cols-10-backg.cols, frame.cols-10));

	//50x100
	Mat backg2(50, 150, CV_8UC3, Scalar(255,255,255));
	backg2.copyTo(frame.rowRange(10, backg2.rows+10).colRange(10, backg2.cols+10));

	Scalar black(0,0,0);

	putText(frame, "Sac Le", Point(20, 45), 3, 1,
				black, 2, 8, false);
	putText(frame, "Dem xe may", Point(frame.cols - 240, 40), 3, 1,
			black, 2, 8, false);
	line(frame, Point(frame.cols - 240, 50), Point(frame.cols - 20, 50),
			black, 2, 2, false);

	string infoText = "Thay [";
	string items = "] xe(s)";
	string sizeMotor = SSTR(motors.size());

	infoText = infoText + sizeMotor + items;

	putText(frame, infoText.c_str(), Point(frame.cols - 200, 70), 3, 0.5,
			black, 1, 8, false);

	String countStr = "Tong so : ";
	String totalStr = SSTR(total);

	countStr = countStr + totalStr;

	putText(frame, countStr.c_str(), Point(frame.cols - 200, 90), 3, 0.5,
			black, 1, 8, false);


	outputVideo << frame;
	//-- Show what you got
	imshow(window_name, frame);
}

string toleadstr(int val, int length){
	stringstream ss;
	ss << setw(length) << setfill('0') << val;
	return ss.str();
}
