/*
 * DHMProcessor.cuh
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <boost/range.hpp>
#include <boost/filesystem.hpp>

#include "DHMCommon.cuh"
#include "ops.cuh"

class DHMProcessor
{
private:
    void gen_filter_quadrant(complex *);
    void transfer_filter_async(complex *, complex *);
    void save_image(const byte *);
    void convert_image(const byte *, complex *);
    void fft_image(complex *);
    void apply_filter(complex *, const complex *, const byte *);
    void ifft_stack(complex *, const byte *);
    void mod_stack(const complex *, float *, const byte *);

    void display_image(byte *);
    void display_volume(float *);

    // experimental parameters

//    std::string inputDir;
    std::string outputDir;

    // TODO: take some of these from constructor
    const int N = 1024;
    const int NUM_SLICES = 100;
    const int NUM_FRAMES = 10;
    const float DX = (5.32f / 1024.f);
    const float DY = (6.66f / 1280.f);
    const float DZ = 1.f;
    const float Z0 = 30.f;
    const float LAMBDA0 = 0.000488f;

    DHMParameters p;

    cv::VideoWriter writer;

    // CUDA stuff

    byte *h_frame, *h_mask;
    byte *d_frame, *d_mask;

    complex *h_filter;
    complex *d_filter[2];
    bool buffer_pos;

    complex *d_image;

    float *h_volume;
    float *d_volume;

    cufftHandle fft_plan;
    const cudaDataType fft_type = CUDA_C_32F;
    long long fft_dims[2] = {N, N};
    size_t fft_work_size = 0;

    cudaStream_t math_stream, copy_stream;

public:
    DHMProcessor(std::string);
    ~DHMProcessor();

    void process_camera();
    void process_folder(std::string);

    void process_frame(byte *, float *, bool, bool);

//    void save_frame(byte *);

    // should this be in constructor?
    void set_callback(void (*)(float *, byte *, void *));
};

