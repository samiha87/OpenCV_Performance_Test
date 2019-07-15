# Remove old log
rm analysis_log
# Run static code analysis tool, outputs errors and warnings to codereview.txt
cppcheck --enable=all --std=c++11 --language=c++ main.cpp sources/* headers/* --output-file=codereview.txt
# Check comment ratio
cloc headers/ sources/ >> analysis_log
# Run program to check the code ratio
# Store results from pythong program, python <program> <input> <allowed comment/code ratio>
result=$(python analyse.py -i "analysis_log" -t "0.3")
if [ "$result" = "Ok" ]; then
  # Run cmake
  cmake .
  # Clean previous make
  make clean
  # Make files
  make
  # Run doxygen and generato documentation
  doxygen Doxyfile
  echo " "
  echo " "
  echo "Code analysis:"
  cat codereview.txt
else
  echo "Failed ratio is lower"
  cat analysis_log
fi
