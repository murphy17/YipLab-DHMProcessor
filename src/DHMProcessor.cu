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

    // allocate buffers, setup FFTs

    CUDA_CHECK( cudaStreamCreate(&math_stream) );
    CUDA_CHECK( cudaStreamCreate(&copy_stream) );

    CUDA_CHECK( cufftCreate(&fft_plan) );
    CUDA_CHECK( cufftXtMakePlanMany(fft_plan, 2, fft_dims,
                                    NULL, 1, 0, fft_type,
                                    NULL, 1, 0, fft_type,
                                    1, &fft_work_size, fft_type) );
    CUDA_CHECK( cufftSetStream(fft_plan, math_stream) );

    // only one quadrant
//    CUDA_CHECK( cudaMallocHost(&filter_stack, (N/2+1)*(N/2+1)*sizeof(complex)) );

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

    // construct PSF, keep on host
    gen_filter_quadrant(filter_stack);
}

DHMProcessor::~DHMProcessor() {
    CUDA_CHECK( cudaFreeHost(filter_stack) );
}

void DHMProcessor::process_camera() {
    // stub

    const int codec = CV_FOURCC('F','F','V','1');
    const int fps = 1;

    const std::time_t now = std::time(0);
    const std::tm ltm = std::localtime(&now)

    std::string path = outputDir + "/" +
                       std::to_string(1900 + ltm->tm_year) + "_" +
                       std::to_string(1 + ltm->tm_mon) + "_" +
                       std::to_string(ltm->tm_mday) + "_" +
                       std::to_string(1 + ltm->tm_hour) + "_" +
                       std::to_string(1 + ltm->tm_min) + "_" +
                       std::to_string(1 + ltm->tm_sec) +
                       ".mpg";

    const int size[2] = {N, N};
    writer.open(path, codec, fps, size, false);
    if (!writer.isOpened()) throw DHMException("Cannot open output video", __LINE__, __FILE__);
    // async
    // also needs to save images
}

void DHMProcessor::process_folder() {
    using namespace boost::filesystem;

    float *volume = new float[NUM_SLICES*N*N];
//        const int vol_sizes[3] = {NUM_SLICES, N, N};

    // loop thru files in input folder
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

void DHMProcessor::process_frame(byte *frame, float *volume, bool camera) {
    // fun part goes here
    // this would be a callback

    // if recording images live, save the raw video feed too
    if (camera)
    {
        // this could happen in another thread, don't think it'll take that long though
        cv::Mat frame_mat(N, N, CV_8U, frame);
        writer.write(frame_mat);
    }

    // convert 8-bit image to real channel of complex float
    ops::_b2c<<<N, N, 0, math_stream>>>(frame, image);
    KERNEL_CHECK();

    // FFT image in-place
    CUDA_CHECK( cufftXtExec(fft_plan, image, image, CUFFT_FORWARD) );

    // multiply image with stored quadrant of filter stack
    ops::_quad_mul<<<N/2+1, N/2+1, 0, math_stream>>>(stack, image, mask, p);
    KERNEL_CHECK();

    // inverse FFT the product, and take complex magnitude
    for (int i = 0; i < NUM_SLICES; i++)
    {
        if (mask[i])
        {
            CUDA_CHECK( cufftXtExec(fft_plan, stack + i*N*N, stack + i*N*N, CUFFT_INVERSE) );

            ops::modulus<<<N, N, 0, math_stream>>>(stack + i*N*N, volume + i*N*N);
            KERNEL_CHECK();
        }
    }

}

//}





