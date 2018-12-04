cmake_minimum_required(VERSION 3.12)
enable_language(C)
SET(CMAKE_CXX_FLAGS -pthread)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(cpp SparseMatrix.cpp SparseMatrix.h SDCA.cpp SDCA.h CriteoSDCATFPreprocess.cpp cirrus/src/SparseDataset.h
        cirrus/src/SparseDataset.cpp cirrus/src/Matrix.h cirrus/src/Matrix.cpp cirrus/src/InputReader.h
        cirrus/src/InputReader.cpp cirrus/src/Dataset.h cirrus/src/Dataset.cpp cirrus/src/Utils.h cirrus/src/Utils.cpp
        cirrus/src/Configuration.h cirrus/src/Configuration.cpp cirrus/src/Checksum.h cirrus/src/Checksum.cpp
        cirrus/src/MurmurHash3.h cirrus/src/MurmurHash3.cpp TFPreprocess.cpp TFPreprocess.h)