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

#include "cufftXt.h"

#include "common.h"

////////////////////////////////////////////////////////////////////////////////
// Error handling
////////////////////////////////////////////////////////////////////////////////

#define DHM_ERROR(msg) (throw DHMException(msg, __FILE__, __LINE__))
class DHMException : public std::exception {
public:
    std::string msg, file;
    int line;

    DHMException(const char *msg, const char *const file, int const line) {
        this->msg = std::string(msg);
        this->line = line;
        this->file = std::string(file);
    }

    const char* what() const throw() {
        cudaDeviceReset();
        std::string str = "DHM exception at line " + std::to_string(line) + " in " + file + ": " + msg;
        return str.c_str();
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

#define KERNEL_CHECK() _check_kernel(__LINE__, __FILE__);
inline void _check_kernel(const int line, const char *file) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw DHMException("kernel failure", file, line);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Callback class for volume processing
////////////////////////////////////////////////////////////////////////////////

//class DHMCallback {
//private:
//    void (*_func)(float *, byte *, void *);
//    void *_out, *_params;
//
//public:
//    DHMCallback(void *, void (*)(float *, byte *, void *), void *);
//    void operator()(float *, byte *);
//};

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
// CUDA timer
////////////////////////////////////////////////////////////////////////////////

#define CUDA_TIMER( expr ) _cuda_timer(0, NULL); (expr); _cuda_timer(1, #expr);
inline void _cuda_timer(const int state, const char *msg)
{
    static cudaEvent_t _timerStart, _timerStop;

    if (state == 0)
    {
        cudaEventCreate(&_timerStart);
        cudaEventCreate(&_timerStop);
        cudaEventRecord(_timerStart);
    }
    else
    {
        float ms;
        cudaEventRecord(_timerStop);
        cudaEventSynchronize(_timerStop);
        cudaEventElapsedTime(&ms, _timerStart, _timerStop);
        std::cout << std::string(msg) << " took " << ms << "ms" << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err)
        {
            std::cout << "Error code " << err << std::endl;
        }
        cudaEventDestroy(_timerStart);
        cudaEventDestroy(_timerStop);
        cudaDeviceSynchronize();
    }
}
