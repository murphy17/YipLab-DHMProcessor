/*
 * DHMProcessor.cuh
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#pragma once

#include "DHMCommon.cuh"

class DHMProcessor
{
public:
    void gen_filter_quadrant(complex *psf);
    void fft_image(const byte *image, complex *image_f);
    void apply_filter(complex *stack, const complex *image, const byte *mask);

    // experimental parameters

    std::string inputDir;
    std::string outputDir;

    const int N = 1024;
    const int NUM_SLICES = 100;
    const int NUM_FRAMES = 10;
    const float DX = (5.32f / 1024.f);
    const float DY = (6.66f / 1280.f);
    const float DZ = 1.f;
    const float Z0 = 30.f;
    const float LAMBDA0 = 0.000488f;

    DHMParameters p;

    // CUDA stuff

    complex *filter_stack; // host

    cufftHandle fft_plan;
    const cudaDataType fft_type = CUDA_C_32F;
    const long long fft_dims[2] = {N, N};
    size_t fft_work_size = 0;

    cudaStream_t math_stream, copy_stream;

    DHMProcessor(std::string inputDir, std::string outputDir);
    ~DHMProcessor();

    void process_camera();
    void process_folder();

    void process_frame(byte *frame, float *volume);
};

