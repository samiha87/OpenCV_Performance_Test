// This Module reads as many images as can and converts them edge drawing all images stored to pc and converting speed measured
#include "defines.h"
#include "image_test.h"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/core/utility.hpp>
#include "iostream"
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <ctime>
#include <chrono>
#include <fstream>
#include <ctime>

#define MAX_IMAGE 10

using namespace cv;
using namespace std;
using namespace std::chrono;


auto startImage = high_resolution_clock::now();

void ImageTest::startTimer() {
	startImage = high_resolution_clock::now();
}

void ImageTest::logTime(std::string msg) {
	ofstream logger;
	// Take current time and set pointer for ctime
	time_t now = time(0);
	// Convert time value to string
	char *dt = ctime(&now);
	// Open log.txt file and append to it, if not exist creates new log.txt
	logger.open("log.txt", ios::app);
	// Stream to log.txt
	logger << dt << msg << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-startImage).count() << "ms\n\r";
	// Close file
	logger.close();
}

void ImageTest::runCPU(void) {
	// Store to log what test and when it's started
	logTime("CPU Image test started");
	// Define Matrix for source file
	Mat src1;
	int i = 0;
	// Define time variables
	double ticker1 = 0;
	double ticker2 = 0;
	double tickerArray[10000];
	int tickPoll = 0;
	double avg = 0.0;

	// Take timestamp of time now
	startTimer();
	// Read MAX_IMAGE amount of files once reached stop
  while(i <= MAX_IMAGE) {
		// Get timestamp
		ticker1 = cv::getTickCount();
		// Convert iterator I from integer to string
		std::string filler = std::to_string(i);
		std::ostringstream oss;
		// Stream to stringstream filename to load
		oss << "images/sample" << filler << ".jpg";
		// Convert stringstream to string
		std::string load_image = oss.str();
		// Load RGB image to matrix src1
		src1 = imread(load_image, cv::IMREAD_COLOR);
		// Define matrixes for conversion
		Mat gray, edge, draw;
		// Convert loaded image to grayscale
		cvtColor(src1, gray, cv::COLOR_BGR2GRAY);
		// Do canny conversion for grayscale image, converts image to edge format
		// Shows image edges as white and rest as black
		Canny( gray, edge, 50, 150, 3);
		// Convert to unsigned 8bit/pixel, can have values 0 - 255
		edge.convertTo(draw, CV_8U);
		std::ostringstream output;
		// Set output file path and name, add iterator i as file number.
		// Store as jpg
		output << "images_processed/processed_cpu_" << filler << ".jpg";
		std::string store_image = output.str();
		// Store converted image from draw matrix to file store_image
		// Saves file as defined on output stream
		cv::imwrite(store_image, draw);
		i++;
		// Get latest clock
		ticker2 = cv::getTickCount();
		tickerArray[tickPoll] = (ticker2 - ticker1) / cv::getTickFrequency();
		double sum = 0;
		// Calculate avarage processing time for image
		for(int ix = 0; ix <= tickPoll; ix++) {
			sum = sum + tickerArray[ix];
		}
		avg = sum / (tickPoll + 1);
		cout << "Calculating " << sum << " / " << tickPoll;
		cout << "AVG performance is " << avg;
		cout << " / Latest processing time " << tickerArray[tickPoll]  << "\n\r";
		tickPoll++;

	}
	// Convert calculated processing time to string format
	std::string filler = std::to_string(avg);
	std::ostringstream oss;
	oss << "CPU Image test: " << filler << "/Frame";
	// One test done, log processing speed to log.txt
	logTime(oss.str());
	// Destroy all open windows
	destroyAllWindows();
  return;
}

void ImageTest::runGPUCuda(void) {
	logTime("GPU() Cuda Image test started");
	Mat src1;
	int i = 0;
	// Timers
	double ticker1 = 0;
	double ticker2 = 0;
	double tickerArray[10000];
	int tickPoll = 0;
	double avg = 0.0;

	startTimer();
  while(i <= MAX_IMAGE) {
		ticker1 = cv::getTickCount();
		std::string filler = std::to_string(i);
		std::ostringstream oss;
		oss << "images/sample" << filler << ".jpg";
		std::string load_image = oss.str();
		src1 = imread(load_image, cv::IMREAD_COLOR);
		cuda::GpuMat gpuMat, gpuDest, gpuDetected, gpuDraw;
		// Store srca1 to GPU
		gpuMat.upload(src1);

		cuda::cvtColor(gpuMat, gpuDest, COLOR_BGR2GRAY);
		//Canny( gray, edge, 50, 150, 3);
		Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cuda::createCannyEdgeDetector(2.0, 100.0, 3, false);
		canny_edg->detect(gpuDest, gpuDetected);
		gpuDetected.convertTo(gpuDraw, CV_8U);
		Mat draw(gpuDraw);
		std::ostringstream output;
		output << "images_processed/processed_" << filler << ".jpg";

		std::string store_image = output.str();
		cv::imwrite(store_image, draw);
		i++;
		// Get latest clock
		ticker2 = cv::getTickCount();
		tickerArray[tickPoll] = (ticker2 - ticker1) / cv::getTickFrequency();
		double sum = 0;
		// Calculate avarage processing time
		for(int ix = 0; ix <= tickPoll; ix++) {
			sum = sum + tickerArray[ix];
		}
		avg = sum / (tickPoll + 1);
		cout << "Calculating " << sum << " / " << tickPoll;
		cout << "AVG performance is " << avg;
		cout << " / Latest processing time " << tickerArray[tickPoll]  << "\n\r";
		tickPoll++;

	}
	std::string filler = std::to_string(avg);
	std::ostringstream oss;
	oss << "CUDA GPU Image test: " << filler << "/Frame";
	logTime(oss.str());
	destroyAllWindows();
  return;
}

void ImageTest::runGPUOpenCL(void) {
	logTime("GPU OpenCL Image test started");
	Mat src1;
	int i = 0;
	// Timers
	double ticker1 = 0;
	double ticker2 = 0;
	double tickerArray[10000];
	int tickPoll = 0;
	double avg = 0.0;

	startTimer();
  while(i <= MAX_IMAGE) {
		ticker1 = cv::getTickCount();
		std::string filler = std::to_string(i);
		std::ostringstream oss;
		oss << "images/sample" << filler << ".jpg";
		std::string load_image = oss.str();
		src1 = imread(load_image, cv::IMREAD_COLOR);
		UMat gray, edge, draw;
		cvtColor(src1, gray, cv::COLOR_BGR2GRAY);
		Canny( gray, edge, 50, 150, 3);
		// Convert to unsigned 8bit/pixel, can have values 0 - 255
		edge.convertTo(draw, CV_8U);
		std::ostringstream output;
		output << "images_processed/processed_cuda_" << filler << ".jpg";

		std::string store_image = output.str();
		cv::imwrite(store_image, draw);
		i++;
		// Get latest clock
		ticker2 = cv::getTickCount();
		tickerArray[tickPoll] = (ticker2 - ticker1) / cv::getTickFrequency();
		double sum = 0;
		// Calculate avarage processing time
		for(int ix = 0; ix <= tickPoll; ix++) {
			sum = sum + tickerArray[ix];
		}
		avg = sum / (tickPoll + 1);
		cout << "Calculating " << sum << " / " << tickPoll;
		cout << "AVG performance is " << avg;
		cout << " / Latest processing time " << tickerArray[tickPoll]  << "\n\r";
		tickPoll++;

	}
	std::string filler = std::to_string(avg);
	std::ostringstream oss;
	oss << "GPU OpenCL Image test: " << filler << "/Frame";
	logTime(oss.str());
	destroyAllWindows();
  return;
}
