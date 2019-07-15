#include "video_test.h"
#include "image_test.h"
#include <stdio.h>

void clearLog() {
  remove("log.txt");
}

int main() {
  VideoTest vTest;
  ImageTest iTest;
  // Delete log file
  clearLog();
  /* Image test
    1. Load image
    2. Convert to gray color
    3. Do canny conversion, only show edges
    4. Convert to 8bit/pixel
    5. Store to file
  */
  // CPU
  iTest.runCPU();  // Test image processing
  // CUDA
  iTest.runGPUCuda();
  // OpenCL
  iTest.runGPUOpenCL();

  /*  Video test and people detection speed
  1. Capture frame
  2. Run trough hog detecotr
  3. Draw on detection frame
  4. Display original and detection frame
  */
  // CPU
  vTest.testCPUHOG();
  // Cuda
  vTest.testGPUCudaHOG();
  // OpenCL
  vTest.testGPUOpenCLHOG();

  /*  Video test and frame resizing
  1. Capture frame
  2. Resize
  3. Display original and detection frame
  */
  // CPU
  vTest.testCPUResize();
  // Cuda
  vTest.testGPUCudaResize();
  // GPU
  vTest.testGPUOpenCLResize();

  return 0;
}
