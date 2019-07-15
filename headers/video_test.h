#include <string>
#include <stdio.h>
#include <stdlib.h>
//! A Process speed test for video footage and live stream on CPU, Cuda and OpenCL
/*!
This class contains few different tests to help determining whats best languange for specific task.
All test when processed are stored with timestamp and results to log.txt file in project folder.
*/
class VideoTest {
public:
  //! Runs HOG Descriptor test for cpu
  /*!
  Loads CPU to test how fast CPU can process object detection alghoritms.
  */
  void testCPUHOG(void);
  //! Runs HOG Descriptor test for Cuda
  /*!
  Loads GPU and CPU to test how fast GPU can process object detection alghoritms.
  */
  void testGPUCudaHOG(void);
  //! Runs HOG Descriptor test for OpenCL
  /*!
  Loads GPU and CPU to test how fast GPU can process object detection alghoritms.
  */
  void testGPUOpenCLHOG(void);
  //! Runs resizing test for CPU
  /*!
  Loads CPU to test frame resizing speed
  */
  void testCPUResize(void);
  //! Runs resizing test for Cuda
  /*!
  Loads CPU and GPU to test frame resizing speed
  */
  void testGPUCudaResize(void);
  //! Runs resizing test for OpenCL
  /*!
  Loads CPU and GPU to test frame resizing speed
  */
  void testGPUOpenCLResize(void);
private:
  //! Start timer
  void startTimer();
  //! Take timestamp and store to log.tx
  /*!
    \param std::string, Message to be stored
  */
  void logTime(std::string msg);
  //! Take timestamp and store to log.tx
  /*!
    \param std::string, Message to be stored
  */
  void logPerformance(std::string msg);
};
