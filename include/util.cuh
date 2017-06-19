/*
 * util.cuh
 *
 *  Created on: Jun 19, 2017
 *      Author: michaelmurphy
 */

#pragma once

#include <string>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

////////////////////////////////////////////////////////////////////////////////
// CUDA timer
////////////////////////////////////////////////////////////////////////////////

#define CUDA_TIMER( expr ) _cuda_timer(0, NULL); (expr); _cuda_timer(1, #expr);
inline void _cuda_timer(const int state, const char *msg)
{
    static std::vector<cudaEvent_t> start, stop;

    if (state == 0)
    {
        cudaEvent_t event;
        start.push_back(event);
        stop.push_back(event);
        cudaEventCreate(&start.back());
        cudaEventCreate(&stop.back());
        cudaEventRecord(start.back());
    }
    else
    {
        float ms;
        cudaEventRecord(stop.back());
        cudaEventSynchronize(stop.back());
        cudaEventElapsedTime(&ms, start.back(), stop.back());
        std::cout << std::string(msg) << " took " << ms << "ms" << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err)
        {
            std::cout << "Error code " << err << std::endl;
        }
        cudaEventDestroy(start.back());
        cudaEventDestroy(stop.back());
        start.pop_back();
        stop.pop_back();
        cudaDeviceSynchronize();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Macros for printing
////////////////////////////////////////////////////////////////////////////////

#define _PRINT_GET(_0, _1, _2, _3, NAME, ...) NAME
#define PRINT(...) _PRINT_GET(__VA_ARGS__, _PRINT_3, _PRINT_2, _PRINT_1, _PRINT_0)(__VA_ARGS__)
#define _PRINT_0(val) _PRINT((val), #val, __LINE__, __FILE__);
#define _PRINT_1(val, m) _PRINT((val), #val, __LINE__, __FILE__, m);
#define _PRINT_2(val, m, n) _PRINT((val), #val, __LINE__, __FILE__, m, n);
#define _PRINT_3(val, m, n, p) _PRINT((val), #val, __LINE__, __FILE__, m, n, p);

#define _CUDA_PRINT_GET(_0, _1, _2, _3, NAME, ...) NAME
#define CUDA_PRINT(...) _CUDA_PRINT_GET(__VA_ARGS__, _CUDA_PRINT_3, _CUDA_PRINT_2, _CUDA_PRINT_1, _CUDA_PRINT_0)(__VA_ARGS__)
#define _CUDA_PRINT_0(val) _CUDA_PRINT((val), #val, __LINE__, __FILE__);
#define _CUDA_PRINT_1(val, m) _CUDA_PRINT((val), #val, __LINE__, __FILE__, m);
#define _CUDA_PRINT_2(val, m, n) _CUDA_PRINT((val), #val, __LINE__, __FILE__, m, n);
#define _CUDA_PRINT_3(val, m, n, p) _CUDA_PRINT((val), #val, __LINE__, __FILE__, m, n, p);

template <typename T>
inline void _PRINT(T val, const char *func, const int line, const char *file)
{
    std::cout << func << " = " << val << std::endl;
}
template <typename T>
inline void _PRINT(T *val, const char *const func, int const line, const char *const file, int const m)
{
    std::cout << func << " = [ ";
    for (int i = 0; i < m; i++)
    {
        std::cout << val[i];
        if (i < m-1)
            std::cout << ", ";
        else
            std::cout << " ]";
    }
    std::cout << std::endl;
}
template <typename T>
inline void _PRINT(T *val, const char *const func, int const line, const char *const file, int const m, int const n)
{
    std::cout << func << " = " << std::endl << "[ ";
    for (int i = 0; i < m; i++)
    {
        std::cout << "[ ";
        for (int j = 0; j < n; j++)
        {
            std::cout << val[i*n+j];
            if (j < n-1)
                std::cout << ", ";
            else
                std::cout << " ]";
        }
        if (i < m-1)
            std::cout << std::endl << "  ";
        else
            std::cout << " ]" << std::endl;
    }
}
template <typename T>
inline void _PRINT(T *val, const char *const func, int const line, const char *const file, int const m, int const n, int const p)
{
    for (int k = 0; k < p; k++)
    {
        _PRINT(val + k*m*n, (std::string(func) + " [:, :, " + std::to_string(k) + "]").c_str(), line, file, m, n);
    }
}

template <typename T>
inline void _CUDA_PRINT(T *val, const char *func, const int line, const char *file)
{
    T x;
    cudaMemcpy(&x, val, sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    _PRINT(x, func, line, file);
}
template <typename T>
inline void _CUDA_PRINT(T *val, const char *func, const int line, const char *file, int const m)
{
    T* x = new T[m];
    cudaMemcpy(x, val, m*sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    _PRINT(x, func, line, file, m);
    delete[] x;
}
template <typename T>
inline void _CUDA_PRINT(T *val, const char *func, const int line, const char *file, int const m, int const n)
{
    T* x = new T[m*n];
    cudaMemcpy(x, val, m*n*sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    _PRINT(x, func, line, file, m, n);
    delete[] x;
}
template <typename T>
inline void _CUDA_PRINT(T *val, const char *func, const int line, const char *file, int const m, int const n, int const p)
{
    T* x = new T[m*n*p];
    cudaMemcpy(x, val, m*n*p*sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    _PRINT(x, func, line, file, m, n, p);
    delete[] x;
}

////////////////////////////////////////////////////////////////////////////////
// Macros for showing image
////////////////////////////////////////////////////////////////////////////////

#define SHOW(val, m, n) _SHOW(#val, (val), m, n);
#define CUDA_SHOW(val, m, n) _CUDA_SHOW(#val, (val), m, n);

inline void _SHOW(const char* const name, unsigned char *x, int const m, int const n)
{
    cv::Mat mat(m, n, CV_8U, x);
    cv::namedWindow(name, CV_WINDOW_NORMAL);
    cv::imshow(name, mat);
    cv::waitKey(0);
}
inline void _SHOW(const char* const name, float *x, int const m, int const n)
{
    cv::Mat mat(m, n, CV_32F, x);
    cv::normalize(mat, mat, 1.0, 0.0, cv::NORM_MINMAX, -1);
    cv::namedWindow(name, CV_WINDOW_NORMAL);
    cv::imshow(name, mat);
    cv::waitKey(0);
}

template <typename T>
inline void _CUDA_SHOW(const char* const name, T *x, int const m, int const n)
{
    T *x_ = new T[m*n];
    cudaMemcpy(x_, x, m*n*sizeof(T), cudaMemcpyDeviceToHost);
    _SHOW(name, x_, m, n);
    delete[] x_;
}


