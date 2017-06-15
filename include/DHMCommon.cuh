/*
 * DHMCommon.hpp
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <exception>

#include "cufftXt.h"

#include "common.h"

////////////////////////////////////////////////////////////////////////////////
// Error handling
////////////////////////////////////////////////////////////////////////////////

class DHMException : public std::exception {
public:
    std::string msg, file;
    int line;

    DHMException(const char *msg, const int line, const char *file) {
        this->msg = std::string(msg);
        this->line = line;
        this->file = std::string(file);
    }

    const char* what() const throw() {
        cudaDeviceReset();
        return (std::string("DHM exception at line ") +
                std::to_string(this->line) + " in " + this->file +
                ": " + this->msg).c_str();
    }
};

#define CUDA_CHECK(val) _check_cuda((val), #val, __FILE__, __LINE__)
template<typename T>
inline void _check_cuda(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        throw DHMException(_cudaGetErrorEnum(result), __LINE__, __FILE__);
    }
}

#define KERNEL_CHECK() _check_kernel(__LINE__, __FILE__);
inline void _check_kernel(const int line, const char *file) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw DHMException("kernel failure", line, file);
    }
}

//inline int check_dir(const std::string dir) {
//    struct stat sb;
//    return (!(stat(dir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)));
//}

////////////////////////////////////////////////////////////////////////////////
// Type definitions (i.e. for 16/32-bit precision)
////////////////////////////////////////////////////////////////////////////////

typedef float2 complex;
typedef float real;
typedef unsigned char byte;

////////////////////////////////////////////////////////////////////////////////
// Class to store experimental parameters
////////////////////////////////////////////////////////////////////////////////

typedef struct
{
    int N, NUM_SLICES, NUM_FRAMES;
    float DX, DY, DZ, Z0, LAMBDA0;
}  DHMParameters;
