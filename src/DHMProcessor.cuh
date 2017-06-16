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

// hoping this ends up using 128-byte ops
typedef struct {
    int x, y, z;
    float v;
} COOTuple;

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
    int volume_to_list(float *, COOTuple **);

    // experimental parameters

//    std::string inputDir;
    std::string outputDir;

    // TODO: take some of these from constructor
    static const int N = 1024;
    static const int NUM_SLICES = 100;
    static const int NUM_FRAMES = 10;
    static constexpr float DX = (5.32f / 1024.f);
    static constexpr float DY = (6.66f / 1280.f);
    static constexpr float DZ = 1.f;
    static constexpr float Z0 = 30.f;
    static constexpr float LAMBDA0 = 0.000488f;
    static constexpr float ZERO_THR = 1e-3;

    static const bool UNIFIED_MEM = false; // Jetson

    static bool is_initialized;

    DHMParameters p;

    cv::VideoWriter writer;

    // CUDA stuff

    byte *h_frame, *h_mask;
    byte *d_frame, *d_mask;

    complex *h_filter;
    complex *d_filter[2];
    bool buffer_pos;

    complex *d_image;

    float *d_volume;

    cufftHandle fft_plan;
    const cudaDataType fft_type = CUDA_C_32F;
    long long fft_dims[2] = {N, N};
    size_t fft_work_size = 0;

    cudaStream_t async_stream;

    void (*callback)(float *, byte *, void *) = nullptr;

    void display_image(byte *);
    void display_volume(float *);

    void load_image(std::string);
    void process_frame(bool);
    void save_volume(std::string);

    void load_volume(std::string, float*);

//    void save_frame(byte *);

    // should this be in constructor?
    void set_callback(void (*)(float *, byte *, void *));

public:
    DHMProcessor(std::string);
    ~DHMProcessor();

    void process_camera();
    void process_folder(std::string);
};

