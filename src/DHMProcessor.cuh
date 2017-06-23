/*
 * DHMProcessor.cuh
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#pragma once

#include "DHMCommon.cuh"
//#include "UEyeCamera/UEyeCamera.hpp"
#include "ImageReader.hpp"

namespace YipLab {

// CUDA kernels
__global__ void _b2c(const byte*, complex*);
__global__ void _freq_shift(complex*);
__global__ void _modulus(const complex*, float*);
__global__ void _gen_filter_slice(complex*, const float, const DHMParameters);
__global__ void _quad_mul(complex*, const complex*, const byte*, const DHMParameters);

// misc helper stuff
std::vector<fs::path> iter_folder(fs::path, std::string = "");
fs::path check_dir(fs::path);

class DHMProcessor
{
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

//    void ueye_callback();
    void process(fs::path);
    void load_image(fs::path);
    void save_image(fs::path);
    void generate_volume();
    void save_volume(fs::path);

    void display_image(byte *);
    void display_volume(float *, bool);

    // experimental parameters
    int num_slices;
    float delta_z;
    float z_init;
    DHMMemoryKind memory_kind; // presently this doesn't do that much
    bool do_save_volume;
    fs::path output_dir;

    // internal parameters
    static bool is_initialized; // singleton
    DHMParameters params;
    DHMCallback callback;
    bool is_running = false;
    int frame_num = 0;
    // std::thread save_thread;

    // CUDA handles
    byte *h_frame, *h_mask, *d_frame, *d_mask;
    complex *h_filter, *d_image;
    complex *d_filter[2]; // double buffering
    float *d_volume, *d_result;
    bool buffer_pos;
    cudaStream_t async_stream;
    cudaMemcpy3DParms memcpy3d_params;
    cufftHandle fft_plan; // CUFFT
    const cudaDataType fft_type = CUDA_C_32F;
    long long fft_dims[2] = {N, N};
    size_t fft_work_size = 0;
//    cusparseHandle_t handle = 0; // CUSparse
//    cusparseMatDescr_t descr = 0;

public:
    DHMProcessor(const int, const float, const float);
    ~DHMProcessor();

    void process_folder(fs::path, fs::path, bool, int = -1);

    // should this be in constructor?
    void set_callback(DHMCallback);

    // TODO: take some of these from constructor
    static const int N = 1024;
    static constexpr float DX = 0.0051992f; //(5.32f / 1024.f);
    static constexpr float DY = 0.0051992f; // (6.66f / 1280.f);
    static constexpr float LAMBDA0 = 0.000488f;
};

}
