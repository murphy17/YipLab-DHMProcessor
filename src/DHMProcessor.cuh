/*
 * DHMProcessor.cuh
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#pragma once

#include "DHMCommon.cuh"
#include "ops.cuh"

namespace YipLab {

class DHMProcessor
{
private:
    void setup_cuda();
    void cleanup_cuda();

    void build_filter_stack();

    void transfer_filter_async(complex *, complex *);
    void save_image(const byte *);
    void convert_image(const byte *, complex *);
    void fft_image(complex *);
    void apply_filter(complex *, const complex *, const byte *);
    void ifft_stack(complex *, const byte *);
    void mod_stack(const complex *, float *, const byte *);

    // experimental parameters
    int num_slices;
    float delta_z;
    float z_init;
    DHMMemoryKind memory_kind;

    static bool is_initialized; // singleton

    DHMParameters p;
    DHMCallback callback;

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

    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    void display_image(byte *);
    void display_volume(float *);

    void load_image(std::string);
    void process_frame();
    void save_volume(std::string);

    void load_volume(std::string, float*);

//    void save_frame(byte *); // TODO

public:
    DHMProcessor(const int, const float, const float);
    DHMProcessor(const int, const float, const float, const DHMMemoryKind);
    ~DHMProcessor();

    void process_camera();
    void process_folder(std::string, std::string);

    void view_volume(std::string);

    // should this be in constructor?
    void set_callback(DHMCallback);

    // TODO: take some of these from constructor
    static const int N = 1024;
    static constexpr float DX = (5.32f / 1024.f);
    static constexpr float DY = (6.66f / 1280.f);
    static constexpr float LAMBDA0 = 0.000488f;
};

}
