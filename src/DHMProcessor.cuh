/*
 * DHMProcessor.cuh
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#pragma once

#include "DHMCommon.cuh"
//#include "UEyeCamera/UEyeCamera.hpp"
#include "ImageQueue.hpp"

namespace YipLab {

// CUDA kernels
__global__ void _b2c(const byte*, complex*);
__global__ void _freq_shift(complex*);
__global__ void _modulus(const complex*, float*);
__global__ void _gen_filter_slice(complex*, const float, const DHMParameters);
__global__ void _quad_mul(complex*, const complex*, const byte*, const DHMParameters);

// misc helper stuff
fs::path check_dir(fs::path);

class DHMProcessor
{
public:
    DHMProcessor(const int, const float, const float, const float, const float, const float);
    ~DHMProcessor();

    void process_folder(fs::path, fs::path, bool, int = -1);

    // should this be in constructor?
    // void set_callback(DHMCallback);

    static const int N = 1024;
    float DX, DY, LAMBDA0;

private:
    void setup_cuda();
    void cleanup_cuda();

    void build_filter_stack();

    void save_image(const byte *);
    void convert_image(const byte *, complex *);
    void fft_image(complex *);
    void apply_filter(complex *, const complex *, const byte *);
    void ifft_stack(complex *, const byte *);
    void mod_stack(const complex *, float *, const byte *);

    void process();

    // experimental parameters
    int num_slices;
    float delta_z;
    float z_init;
    DHMMemoryKind memory_kind; // presently this doesn't do that much
    fs::path output_dir;

    // internal parameters
    static bool is_initialized; // singleton
    DHMParameters params;
//    DHMCallback callback;
    bool is_running = false;
    int frame_num = 0;
    static const int N_BUF = 2;
    static const int QUEUE_SIZE = 16;

    // CUDA handles
    byte *h_mask, *d_frame, *d_mask;
    complex *h_filter, *d_image;
    complex *d_filter[N_BUF]; // multiple buffering
    float *d_volume, *d_depth;
    float *h_depth;
    int buffer_pos;
    cudaStream_t stream[N_BUF];
    cudaMemcpy3DParms memcpy3d_params;
    cufftHandle fft_plan; // CUFFT
    const cudaDataType fft_type = CUDA_C_32F;
    long long fft_dims[2] = {N, N};
    size_t fft_work_size = 0;
};

}
