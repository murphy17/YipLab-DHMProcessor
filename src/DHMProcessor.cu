/*
 * DHMProcessor.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: michaelmurphy
 */

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <boost/range.hpp>
#include <boost/filesystem.hpp>

#include "DHMCommon.cuh"
#include "DHMProcessor.cuh"

//namespace YipLab {

DHMProcessor::DHMProcessor(std::string inputDir, std::string outputDir) {
    this->inputDir = inputDir;
    this->outputDir = outputDir;

    // reset the GPU, use proper exceptions to do this...
    CUDA_CHECK( cudaDeviceReset() );

    // camera crap would go here...

    // make sure input, output directories are fine
    using namespace boost::filesystem;
    if ( !is_directory(inputDir) ) throw DHMException("Input directory not found", __LINE__, __FILE__);
    if ( !is_directory(outputDir) ) throw DHMException("Output directory not found", __LINE__, __FILE__);

    // pack parameters
    p = { N, NUM_SLICES, NUM_FRAMES, DX, DY, DZ, Z0, LAMBDA0 };

    // allocate buffers, setup FFT

    // what a fucking mess
//        complex *image;
//        checkCudaErrors( cudaMalloc(&image, N*N*sizeof(complex)) );
//        complex *psf;
//        checkCudaErrors( cudaMalloc(&psf, N*N*sizeof(complex)) );
//
//        complex *host_psf;
//        checkCudaErrors( cudaMallocHost(&host_psf, NUM_SLICES*(N/2+1)*(N/2+1)*sizeof(complex)) );
//
//        byte *image_u8;
//        checkCudaErrors( cudaMalloc(&image_u8, N*N*sizeof(byte)) );
//
//        cudaStream_t math_stream, copy_stream;
//        checkCudaErrors( cudaStreamCreate(&math_stream) );
//        checkCudaErrors( cudaStreamCreate(&copy_stream) );
//
//        complex *in_buffers[2];
//        checkCudaErrors( cudaMalloc(&in_buffers[0], NUM_SLICES*N*N*sizeof(complex)) );
//        checkCudaErrors( cudaMalloc(&in_buffers[1], NUM_SLICES*N*N*sizeof(complex)) );
//
//        real *out_buffer;
//        checkCudaErrors( cudaMalloc(&out_buffer, NUM_SLICES*N*N*sizeof(real)) );
//
//        // managed memory would be much nicer here, esp on Tegra, but was causing problems w/ streams
//        char *host_mask, *mask;
//        // checkCudaErrors( cudaMallocManaged(&mask, NUM_SLICES*sizeof(char), cudaMemAttachGlobal) );
//        checkCudaErrors( cudaMallocHost(&host_mask, NUM_SLICES*sizeof(char)) );
//        checkCudaErrors( cudaMalloc(&mask, NUM_SLICES*sizeof(char)) );
//        memset(host_mask, 1, NUM_SLICES);
//        checkCudaErrors( cudaMemcpy(mask, host_mask, NUM_SLICES*sizeof(char), cudaMemcpyHostToDevice) );

}

void DHMProcessor::process_camera() {
    // stub
    // also needs to save images
}

void DHMProcessor::process_folder() {
    using namespace boost::filesystem;

    float *volume = new float[NUM_SLICES*N*N];
//        const int vol_sizes[3] = {NUM_SLICES, N, N};

    for (auto &f : boost::make_iterator_range(directory_iterator(inputDir), {}))
    {
        std::string path = f.path().string();

        // make sure the image is NxN? make sure it's an image / bitmap?
        if (path.substr(path.find_last_of(".") + 1) != "bmp")
            throw DHMException("Files must be bitmaps", __LINE__, __FILE__);

        cv::Mat frame_mat = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
        if ( frame_mat.cols != N || frame_mat.rows != N )
            throw DHMException("Images must be of size NxN", __LINE__, __FILE__);

        byte *frame = frame_mat.data;

        process_frame(frame, volume); // callback!!!

        // write volume to disk... what format? HDF5?
    }
}

void DHMProcessor::process_frame(byte *frame, float *volume) {
    // fun part goes here
    // this would be a callback
}

//}





