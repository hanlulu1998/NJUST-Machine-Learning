cmake_minimum_required(VERSION 3.17)
project(Project1)

set(CMAKE_CXX_STANDARD 11)

include_directories("D:/CPPLibs/Armadillo/include")

link_directories("D:/CPPLibs/Armadillo/lib")

link_libraries("D:/CPPLibs/Armadillo/lib/blas_win64_MT.lib"
        "D:/CPPLibs/Armadillo/lib/lapack_win64_MT.lib")

add_executable(Project1 project1Main.cpp)
