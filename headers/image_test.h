//! Process speed test class for images.
/*
Image test class contains tests for cpu, Cuda and OpenCl.
Tested process goes as follows.
1. Load image
2. Convert to gray color
3. Do canny conversion, only show edges
4. Convert to 8bit/pixel
5. Store to file
*/
#include <string>
#include <stdio.h>
#include <stdlib.h>

class ImageTest {
public:

  //! Runs process speed test for CPU
  void runCPU(void);
  //! Runs process speed test for Cuda
  void runGPUCuda(void);
  //! Runs process speed test for OpenCL
  void runGPUOpenCL(void);
private:
  //! Start timer
  void startTimer();
  //! Take timestamp and store to log.tx
  /*!
    \param std::string, Message to be stored
  */
  void logTime(std::string msg);

};
