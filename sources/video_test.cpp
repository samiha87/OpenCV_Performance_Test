//Uncomment the following line if you are compiling this code in Visual Studio
//#include "stdafx.h"
#include "defines.h"
#include "video_test.h"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/core/utility.hpp>

#include <iostream>
#include <ctime>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>

using namespace cv;
using namespace std;
using namespace std::chrono;

auto start = high_resolution_clock::now();

// Hod detect multiscale(InputArray, Return vector, hit threshold, wundiw strude, padding, scale, final threshold)
//gpuHog->detectMultiScale(detectionLayer, found, 0, Size(8,8), Size(32,32), 1.05, 2);

Size win_stride(8, 8);
double scale = 1.05;
int nlevels = 13;
int gr_threshold = 8;
double hit_threshold = 1.4;

void VideoTest::startTimer() {
        start = high_resolution_clock::now();
}

void VideoTest::logTime(std::string msg) {
  ofstream logger;
  time_t now = time(0);
  char *dt = ctime(&now);

  logger.open("log.txt", ios::app);
  logger << dt << msg << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count() << "ms\n\r";
  logger.close();
}

void VideoTest::logPerformance(std::string msg) {
  ofstream logger;
  time_t now = time(0);
  char *dt = ctime(&now);

  logger.open("log.txt", ios::app);
  logger << dt << msg << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count() << "ms\n\r";
  logger.close();
}
//--------------------------------------------------------------
// People detection test on video using CPU
void VideoTest::testCPUHOG(void) {
    // Log that we starting CPU HOG test
    logPerformance("VideoTest::testCPU() People detector");
    cout << "VideoTest::testCPU()"<< endl;
    // Define timer variables
    double ticker1 = 0;
    double ticker2 = 0;
    double tickerArray[10000];
    int tickPoll = 0;
    double avg = 0.0;
    // Set names for windows displaying frames
    String window = "Detector people";
    String window_original = "Original";
    // Create and define windows
    namedWindow(window_original, WINDOW_NORMAL);
    namedWindow(window, WINDOW_NORMAL);
    // Capture video
    VideoCapture cap("images/sample.mp4");
    // If video couldn't be captured close this function
    if (cap.isOpened() == false) {
     cout << "Cannot open the video file" << endl;
     return;
    }
    // Create the HOG descriptor and detector with default params
    HOGDescriptor hog;
    // Set coefficients for the linear SVM classifier, set coefficients for trained people detection
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());


    while (true) {
      // Store start time, ticker1 contains start time from frame processing
      ticker1 = cv::getTickCount();
      // Define CPU matrix
      Mat frame;
      // Load sample video
      // Get new frame from camera
      bool bSuccess = cap.read(frame); // read a new frame from video
      // If end of video is found exit while loop
      if( bSuccess == false) {
        cout << "Found the end of the video" << endl;
        break;
      }
      // Generate new matrix to display detectection image. Frame contains original frames
      Mat detectionLayer = frame.clone();;
      vector<Rect> found, found_filtered;
      // Detect objects of different sizes in the input image. Return objects as a list of rectangles
      //hog.detectMultiScale(detectionLayer, found, 1.4, Size(8, 8), Size(0, 0), 1.05, 8);
      // Hod detect multiscale(InputArray, Return vector, hit threshold, wundiw strude, padding, scale, final threshold)
      //hog.detectMultiScale(detectionLayer, found, 0, Size(8, 8), Size(0, 0), 1.05, 8);
      hog.detectMultiScale(detectionLayer, found, hit_threshold, win_stride, Size(0,0), scale, gr_threshold);
      // If person detected, draw on detection detectionLayer
      size_t i, j;
      // Cach start time for performance measurement
      time_t now = time(0);
      // Found contains vector for each found rectangles
      //
      for (i = 0; i < found.size(); i++) {
        // Pull rectangle from container
        Rect r = found[i];
        // Loop through container size until
        for (j = 0; j < found.size(); j++) {
          /*
            12 = 00001100 (In Binary)
            25 = 00011001 (In Binary)

            Bit Operation of 12 and 25
              00001100
            & 00011001
            ________
              00001000  = 8 (In decimal)

              00010101
            & 00010101
            ––––––––––
              00010101

              00010101 // r
            & 11110101 // found[j]
            ––––––––––
              00010101 // r
          */
          // Break if j and i ain't same, and binaries of vectir r and found[j] are equal.
          if (j != i && (r & found[j]) == r) break;
        }
        if (j == found.size()) {
          // Pushes r as last element of found_filtered.
          // Increases the container size by one
          found_filtered.push_back(r);
        }
      }
      // Loop through all filtered rectangles and draw them on detectionLayer.
      // multiply detected vector with float and round it to nearest integer value(cvRound)
      for (i = 0; i < found_filtered.size(); i++) {
        // Create rectangle structure
        Rect r = found_filtered[i];
        // Round rectangle value and multiply with 0.1
        r.x += cvRound(r.width*0.1);
        // Round witdth of rectangle and multiply by 0.8 to fit the frame scale
	      r.width = cvRound(r.width*0.8);
        // Set rectangle start point on y axis
	      r.y += cvRound(r.height*0.06);
        // Set rectangle height and round the value
	      r.height = cvRound(r.height*0.9);
        // Draw rectangle to frame/layer
	      rectangle(detectionLayer, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
	    }
      // Get latest clock
      ticker2 = cv::getTickCount();
      // Store to array time spent processing one frame
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
      // Display detectection layer for user
      imshow(window, detectionLayer);
      imshow(window_original, frame);
      // Wait for 1ms or if esc is pressed stop video processing
      if (waitKey(1) == 27) {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        break;
      }
    }

    std::string filler = std::to_string(avg);
    std::ostringstream oss;
    oss << "Video performance: " << filler << "/Frame";
    logPerformance(oss.str());
    destroyAllWindows();
}

// People detection test on video using GPU
void VideoTest::testGPUCudaHOG(void) {
    logPerformance("VideoTest::testGPU() CUDA People detector");
		if (cv::cuda::getCudaEnabledDeviceCount() == 0) {

			cout << "No GPU found or the library is compiled without CUDA support" << endl;
			return;
		}
		cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    cout << "VideoTest::testGPU()"<< endl;
    double ticker1 = 0;
    double ticker2 = 0;
    double tickerArray[10000];
    int tickPoll = 0;
    double avg = 0.0;
    String window = "Detector people";
    String window_original = "Original";
    namedWindow(window_original, WINDOW_NORMAL);
    namedWindow(window, WINDOW_NORMAL);

    VideoCapture cap("images/sample.mp4");

    if (cap.isOpened() == false) {
     cout << "Cannot open the video file" << endl;
     cin.get(); //wait for any key press
     return;
    }
    // Create the GPU accelerated HOG descriptor and detector with default params

    int win_width = 48;
    int win_stride_width = 8;
    int win_stride_height = 8;
    int block_width = 16;
    int block_stride_width = 8;
    int block_stride_height = 8;
    int cell_width = 8;
    int nbins = 9;

    cout << "VideoTEst::TestGPU() CUDA Init cuda hog"<< endl;
    Size win_size(win_width, win_width * 2);
    Size block_size(block_width, block_width);
    Size block_stride(block_stride_width, block_stride_height);
    Size cell_size(cell_width, cell_width);
    //cv::Ptr<cv::cuda::HOG> gpuHog = cv::cuda::HOG::create();
    /*cv::Ptr<cv::cuda::HOG> gpuHog = cv::cuda::HOG::create(win_size, block_size,
      block_stride, cell_size, nbins);
      */
    //cv::Ptr< cv::cuda::HOG> gpuHog;
    cv::Ptr<cv::cuda::HOG> gpuHog = cv::cuda::HOG::create(Size(64, 128),Size(16, 16),Size(8, 8),Size(8, 8),9);
    // Set coefficients for the linear SVM classifier, set coefficients for trained people detection
    cout << "VideoTEst::TestGPU() Set SVM detector"<< endl;
    Mat detector = gpuHog->getDefaultPeopleDetector();
    gpuHog->setSVMDetector(detector);
    cout << "VideoTEst::TestGPU() CUDA Start capturing video "<< endl;
    while (true) {
      ticker1 = cv::getTickCount();
      // Define CPU matrix, img matrix needed to convert fame suitable color format for GPU HOG
      Mat frame, img;
      // Load sample video
      // Get new frame from camera
      bool bSuccess = cap.read(frame); // read a new frame from video
      // If end of video is found exit while loop
      if( bSuccess == false) {
        cout << "Found the end of the video" << endl;
        break;
      }

      // Generate new matrix to display detectection image. Frame contains original frames

      cuda::GpuMat detectionLayer(frame);
      cuda::GpuMat colorConversion;
      cuda::cvtColor(detectionLayer, colorConversion, COLOR_BGR2BGRA);
      // Upload captured image to GPU
      vector<Rect> found, found_filtered;
      // Detect objects of different sizes in the input image. Return objects as a list of rectangles
      // Hod detect multiscale(InputArray, Return vector, hit threshold, wundiw strude, padding, scale, final threshold)
      //gpuHog->detectMultiScale(detectionLayer, found, 0, Size(8,8), Size(32,32), 1.05, 2);
      gpuHog->setNumLevels(nlevels);
      gpuHog->setHitThreshold(hit_threshold);
      gpuHog->setWinStride(win_stride);
      gpuHog->setScaleFactor(scale);
      gpuHog->setGroupThreshold(gr_threshold);
      gpuHog->detectMultiScale(colorConversion, found);

      // If person detected, draw on detection detectionLayer
      size_t i, j;
      // Cach start time for performance measurement
      time_t now = time(0);
      // Found contains vector for each found rectangles
      //
      for (i = 0; i < found.size(); i++) {
        // Pull rectangle from container
        Rect r = found[i];
        // Loop through container size until
        for (j = 0; j < found.size(); j++) {
          /*
            12 = 00001100 (In Binary)
            25 = 00011001 (In Binary)

            Bit Operation of 12 and 25
              00001100
            & 00011001
            ________
              00001000  = 8 (In decimal)

              00010101
            & 00010101
            ––––––––––
              00010101

              00010101 // r
            & 11110101 // found[j]
            ––––––––––
              00010101 // r
          */
          // Break if j and i ain't same, and binaries of vectir r and found[j] are equal.
          if (j != i && (r & found[j]) == r) break;
        }
        if (j == found.size()) {
          // Pushes r as last element of found_filtered.
          // Increases the container size by one
          found_filtered.push_back(r);
        }
      }
      // Loop through all filtered rectangles and draw them on detectionLayer.
      // multiply detected vector with float and round it to nearest integer value(cvRound)
      for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        r.x += cvRound(r.width*0.1);
	      r.width = cvRound(r.width*0.8);
	      r.y += cvRound(r.height*0.06);
	      r.height = cvRound(r.height*0.9);
        // Draw rectangle to frame/layer
	      rectangle(colorConversion, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
	    }
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
      // Display detectection layer for user
      Mat imgToDisplay(colorConversion);
      imshow(window, imgToDisplay);
      imshow(window_original, frame);

      if (waitKey(1) == 27) {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        break;
      }
    }

    std::string filler = std::to_string(avg);
    std::ostringstream oss;
    oss << "Video performance: " << filler << "/Frame";
    logPerformance(oss.str());
    destroyAllWindows();
}

// People detection test on video using OpenCL
void VideoTest::testGPUOpenCLHOG(void) {
    logPerformance("VideoTest::testGPU() OpenCL People detector");
    cout << "VideoTest::testGPU() OpenCL"<< endl;
    double ticker1 = 0;
    double ticker2 = 0;
    double tickerArray[10000];
    int tickPoll = 0;
    double avg = 0.0;
    String window = "Detector people";
    String window_original = "Original";
    namedWindow(window_original, WINDOW_NORMAL);
    namedWindow(window, WINDOW_NORMAL);

    VideoCapture cap("images/sample.mp4");

    if (cap.isOpened() == false)
    {
     cout << "Cannot open the video file" << endl;
     cin.get(); //wait for any key press
     return;
    }
    // Create the HOG descriptor and detector with default params
    HOGDescriptor hog;
    // Set coefficients for the linear SVM classifier, set coefficients for trained people detection
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());


    while (true) {
      ticker1 = cv::getTickCount();
      // Define CPU matrix
      UMat frame;
      // Load sample video
      // Get new frame from camera
      bool bSuccess = cap.read(frame); // read a new frame from video
      // If end of video is found exit while loop
      if( bSuccess == false) {
        cout << "Found the end of the video" << endl;
        break;
      }
      // Generate new matrix to display detectection image. Frame contains original frames
      UMat detectionLayer = frame.clone();;
      vector<Rect> found, found_filtered;
      // Detect objects of different sizes in the input image. Return objects as a list of rectangles
      //hog.detectMultiScale(detectionLayer, found, 1.4, Size(8, 8), Size(0, 0), 1.05, 8);
      // Hod detect multiscale(InputArray, Return vector, hit threshold, wundiw strude, padding, scale, final threshold)
      //hog.detectMultiScale(detectionLayer, found, 0, Size(8, 8), Size(0, 0), 1.05, 8);
      hog.detectMultiScale(detectionLayer, found, hit_threshold, win_stride, Size(0,0), scale, gr_threshold);
      // If person detected, draw on detection detectionLayer
      size_t i, j;
      // Cach start time for performance measurement
      time_t now = time(0);
      // Found contains vector for each found rectangles
      //
      for (i = 0; i < found.size(); i++) {
        // Pull rectangle from container
        Rect r = found[i];
        // Loop through container size until
        for (j = 0; j < found.size(); j++) {
          /*
            12 = 00001100 (In Binary)
            25 = 00011001 (In Binary)

            Bit Operation of 12 and 25
              00001100
            & 00011001
            ________
              00001000  = 8 (In decimal)

              00010101
            & 00010101
            ––––––––––
              00010101

              00010101 // r
            & 11110101 // found[j]
            ––––––––––
              00010101 // r
          */
          // Break if j and i ain't same, and binaries of vectir r and found[j] are equal.
          if (j != i && (r & found[j]) == r) break;
        }
        if (j == found.size()) {
          // Pushes r as last element of found_filtered.
          // Increases the container size by one
          found_filtered.push_back(r);
        }
      }
      // Loop through all filtered rectangles and draw them on detectionLayer.
      // multiply detected vector with float and round it to nearest integer value(cvRound)
      for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        r.x += cvRound(r.width*0.1);
	      r.width = cvRound(r.width*0.8);
	      r.y += cvRound(r.height*0.06);
	      r.height = cvRound(r.height*0.9);
        // Draw rectangle to frame/layer
	      rectangle(detectionLayer, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
	    }
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
      // Display detectection layer for user
      imshow(window, detectionLayer);
      imshow(window_original, frame);

      if (waitKey(1) == 27) {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        break;
      }
    }

    std::string filler = std::to_string(avg);
    std::ostringstream oss;
    oss << "Video performance: " << filler << "/Frame";
    logPerformance(oss.str());
    destroyAllWindows();
}


//..............................................................

//-------------------------------------------------------------
// Frame resize test on CPU
void VideoTest::testCPUResize(void) {
    logPerformance("VideoTest::testCPU() Resize");
    cout << "VideoTest::testCPU() Resize"<< endl;
    double ticker1 = 0;
    double ticker2 = 0;
    double tickerArray[10000];
    int tickPoll = 0;
    double avg = 0.0;
    String window = "Processed";
    String window_original = "Original";
    namedWindow(window_original, WINDOW_NORMAL);
    namedWindow(window, WINDOW_NORMAL);

    VideoCapture cap("images/sample.mp4");

    if (cap.isOpened() == false)
    {
     cout << "Cannot open the video file" << endl;
     cin.get(); //wait for any key press
     return;
    }

    while (true) {
      ticker1 = cv::getTickCount();
      // Define CPU matrix
      Mat frame, processedFrame;
      // Load sample video
      // Get new frame from camera
      bool bSuccess = cap.read(frame); // read a new frame from video
      // If end of video is found exit while loop
      if( bSuccess == false) {
        cout << "Found the end of the video" << endl;
        break;
      }
      resize(frame, processedFrame, Size(200, 100));
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
      // Display detectection layer for user
      imshow(window, processedFrame);
      imshow(window_original, frame);

      if (waitKey(1) == 27) {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        break;
      }
    }

    std::string filler = std::to_string(avg);
    std::ostringstream oss;
    oss << "Video performance: " << filler << "/Frame:";
    logPerformance(oss.str());
    destroyAllWindows();
}

// Frame resize test on GPU
void VideoTest::testGPUCudaResize(void) {
    logPerformance("VideoTest::testGPU() CUDA Resize");
		if (cv::cuda::getCudaEnabledDeviceCount() == 0) {

			cout << "No GPU found or the library is compiled without CUDA support" << endl;
			return;
		}
		cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    cout << "VideoTest::testGPU() Resize"<< endl;
    double ticker1 = 0;
    double ticker2 = 0;
    double tickerArray[10000];
    int tickPoll = 0;
    double avg = 0.0;
    String window = "Processed";
    String window_original = "Original";
    namedWindow(window_original, WINDOW_NORMAL);
    namedWindow(window, WINDOW_NORMAL);

    VideoCapture cap("images/sample.mp4");

    if (cap.isOpened() == false) {
     cout << "Cannot open the video file" << endl;
     cin.get(); //wait for any key press
     return;
    }
    // Create the GPU accelerated HOG descriptor and detector with default params

    while (true) {
      ticker1 = cv::getTickCount();
      // Define CPU matrix, img matrix needed to convert fame suitable color format for GPU HOG
      Mat frame, img;
      // Load sample video
      // Get new frame from camera
      bool bSuccess = cap.read(frame); // read a new frame from video
      // If end of video is found exit while loop
      if( bSuccess == false) {
        cout << "Found the end of the video" << endl;
        break;
      }

      // Generate new matrix to display detectection image. Frame contains original frames
      cuda::GpuMat gpuSrc(frame);
      cuda::GpuMat gpuProcessed;
      cuda::resize(gpuSrc, gpuProcessed, Size(200,100));

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
      // Display detectection layer for user
      Mat imgToDisplay(gpuProcessed);
      imshow(window, imgToDisplay);
      imshow(window_original, frame);

      if (waitKey(1) == 27) {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        break;
      }
    }

    std::string filler = std::to_string(avg);
    std::ostringstream oss;
    oss << "Video performance: " << filler << "/Frame";
    logPerformance(oss.str());
    destroyAllWindows();
}

// Frame resize test on CPU
void VideoTest::testGPUOpenCLResize(void) {
    logPerformance("VideoTest::testGPU() OpenCl Resize");
    cout << "VideoTest::testGPU() OpenCL Resize"<< endl;
    double ticker1 = 0;
    double ticker2 = 0;
    double tickerArray[10000];
    int tickPoll = 0;
    double avg = 0.0;
    String window = "Processed";
    String window_original = "Original";
    namedWindow(window_original, WINDOW_NORMAL);
    namedWindow(window, WINDOW_NORMAL);

    VideoCapture cap("images/sample.mp4");

    if (cap.isOpened() == false)
    {
     cout << "Cannot open the video file" << endl;
     cin.get(); //wait for any key press
     return;
    }

    while (true) {
      ticker1 = cv::getTickCount();
      // Define CPU matrix
      UMat frame, processedFrame;
      // Load sample video
      // Get new frame from camera
      bool bSuccess = cap.read(frame); // read a new frame from video
      // If end of video is found exit while loop
      if( bSuccess == false) {
        cout << "Found the end of the video" << endl;
        break;
      }
      resize(frame, processedFrame, Size(200, 100));
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
      // Display detectection layer for user
      imshow(window, processedFrame);
      imshow(window_original, frame);

      if (waitKey(1) == 27) {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        break;
      }
    }

    std::string filler = std::to_string(avg);
    std::ostringstream oss;
    oss << "Video performance: " << filler << "/Frame:";
    logPerformance(oss.str());
    destroyAllWindows();
}
//.............................................................
