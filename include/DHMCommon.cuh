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
#include <fstream>
#include <exception>
#include <ctime>
#include <algorithm>
#include <vector>
#include <cstdlib>

#include <cusparse_v2.h>
#include <cufftXt.h>

#include "common.h"
#include "util.cuh"

////////////////////////////////////////////////////////////////////////////////
// Error handling
////////////////////////////////////////////////////////////////////////////////

#define DHM_ERROR(msg) (throw DHMException(msg, __FILE__, __LINE__))
class DHMException : public std::exception {
public:
    std::string msg;
//    int line;

    DHMException(const char *msg, const char *const file, int const line) {
        this->msg =  "DHM exception at line " + std::to_string(line) + " in " + file + ": " + msg;
    }

    const char* what() const throw() {
        cudaDeviceReset();
        return msg.c_str();
    }
};

#define CUDA_CHECK(val) _check_cuda((val), #val, __FILE__, __LINE__)
template<typename T>
inline void _check_cuda(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        throw DHMException(_cudaGetErrorEnum(result), file, line);
    }
}

#define KERNEL_CHECK() _check_kernel(__FILE__, __LINE__);
inline void _check_kernel(const char *const file, const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw DHMException(_cudaGetErrorEnum(err), file, line);
    }
}

#define HANDLE_ERROR(err) std::cerr << err.what() << std::endl; exit(-1);

////////////////////////////////////////////////////////////////////////////////
// Type definitions (i.e. for 16/32-bit precision)
////////////////////////////////////////////////////////////////////////////////

typedef float2 complex;
typedef unsigned char byte;

////////////////////////////////////////////////////////////////////////////////
// Class to store experimental parameters
////////////////////////////////////////////////////////////////////////////////

typedef struct
{
    int N, NUM_SLICES, NUM_FRAMES;
    float DX, DY, DZ, Z0, LAMBDA0;
}  DHMParameters;

////////////////////////////////////////////////////////////////////////////////
// Callback class for volume processing
////////////////////////////////////////////////////////////////////////////////

// TODO: function prototype is void callback(float *d_volume, float *d_mask)
// for now I assume volume processing happens *in-place*, and takes *no extra handles or parameters*

// idea here is that you can write a custom processing routine without needing to edit any of my stuff

class DHMCallback {
private:
    void (*_func)(float *, byte *, DHMParameters);

public:
    DHMCallback();
    DHMCallback(void (*)(float *, byte *, DHMParameters));
    void operator()(float *, byte *, DHMParameters);
};
