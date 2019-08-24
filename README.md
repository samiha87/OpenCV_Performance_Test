# OpenCV_Peroformance_Test
Tests OpenCV processing performance for CPU, Cuda, OpenCL and Intel MKL

This program tests performance for OpenCV on different GPU API:s and CPU. 

Performed tests for CPU, Cuda, OpenCL, MKL. 
To test MKL you have to compile opencv with MKL support
To install, install all necessary programs and libraries. 

Necessary programs
Cuda
OpenCL libraries
MKL libraries
OpenCV
cloc              sudo apt-get install cloc
doxygen           sudo apt-get install doxygen
cppcheck          sudo apt-get install cppcheck 
CMAKE

To Run cmake ensure OPENCV_EXTRA_MODULES_PATH is correct. This contains path to opencv_contrib files.
CMAKE instruction for CPU, Cuda and OpenCL 
cmake -D CMAKE_BUILD_TYPE=RELEASE       -D CMAKE_INSTALL_PREFIX=/usr/local       -D INSTALL_C_EXAMPLES=OFF       -D INSTALL_PYTHON_EXAMPLES=OFF       -D WITH_TBB=ON       -D WITH_V4L=ON       -D WITH_QT=ON       -D WITH_OPENGL=ON       -D WITH_GSTREAMER=ON       -D WITH_CUDA=ON       -D WITH_NVCUVID=ON       -D ENABLE_FAST_MATH=1       -D CUDA_FAST_MATH=1       -D WITH_CUBLAS=ON       -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES"       -D BUILD_opencv_cudacodec=OFF       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules       -D BUILD_EXAMPLES=OFF ..
CMAKE instructions for CPU, Cuda, OpenCL and MKL

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_MKL=ON -D MKL_USE_MULTITHREAD=ON -D MKL_WITH_TBB=ON -D WITH_OPENGL=ON -D WITH_GSTREAMER=ON -D WITH_CUDA=ON -D WITH_NVCUVID=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=ON -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D BUILD_opencv_cudacodec=OFF -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D BUILD_EXAMPLES=OFF ..

References
Use Intel® MKL with OpenCV for FFT calculation | Intel® Software
